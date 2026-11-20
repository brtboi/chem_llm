#!/usr/bin/env python3
"""STEP 5: Generate QE input files for all 51 Si structures.

For each structure NNN (000..050):
  calculations/NNN/
    Si.upf          (pseudopotential, copied from template/)
    kpath.json      (k-path: MP's path for 000, seekpath for perturbed)
    pw.in           (SCF input, written by write_espresso_in)
    bands.in        (bands input, written by write_espresso_in)
    bands_post.in   (bands.x input, written by write_fortran_namelist)

Key decisions (see STEP 4 note):
  - ecutwfc = 20.0 Ry  (Pseudo-Dojo recommended 10.0 Ha * 2 Ry/Ha, INVARIANT 5)
  - ecutrho = 160.0 Ry (PAW: 8 * ecutwfc)
  - occupations='fixed' (Si is a clear semiconductor, gap ~0.6 eV)
  - conv_thr = 1.0e-8 Ry
  - nspin = 1 (no magnetism)
  - No spin-orbit (scalar-relativistic pseudopotential, matching MP)
  - No Hubbard U
  - nbnd = 20 (well above 4 occupied bands, covers conduction bands to ~+10 eV)
  - SCF k-mesh: scf_kmesh(prim, spacing=0.2) -- converged by converge_kmesh
  - bands k-points: kp_kpts(kp) from the k-path (INVARIANT 3)
  - bands_post.in: lsym=.false. (skip band-representation file)
  - pseudo_dir = '.' so pw.x finds Si.upf in the current directory (calculations/NNN/)

INVARIANT 1: nat/ntyp computed from the actual ASE Atoms object, never hardcoded.
INVARIANT 3: k-path from chem_llm.dft.kpath, cell = kp_atoms(kp).
INVARIANT 5: cutoffs converted from Ha to Ry explicitly.
"""
import os
import shutil
import numpy as np
from ase.io import read, write
from ase.io.espresso import write_espresso_in, write_fortran_namelist

# --- chem_llm.dft.kpath functions (same code as get_kpath tool) ---
from chem_llm.dft.kpath import (
    standard_kpath, kp_atoms, kp_kpts, kp_save, load_kpath, scf_kmesh,
)

# --- constants ---
N_STRUCTURES = 51
TEMPLATE_DIR = 'template'
STRUCTURES_DIR = 'structures'
CALC_DIR = 'calculations'

# Pseudopotential
PSEUDO_FILE = 'Si.upf'          # filename in template/ and in calculations/NNN/

# Cutoffs (INVARIANT 5: Pseudo-Dojo recommends 10.0 Ha; 1 Ha = 2 Ry)
ECUTWFC_RY = 20.0               # 2 * 10.0 Ha
ECUTRHO_RY = 8.0 * ECUTWFC_RY   # PAW: 8x ecutwfc = 160 Ry

# SCF settings
PREFIX = 'si'
OUTDIR = './tmp'
CONV_THR = 1.0e-8               # Ry
OCCUPATIONS = 'fixed'

# Bands settings
NBND = 20                       # well above 4 occupied bands

# SCF k-mesh spacing (converged by converge_kmesh: 0.2 A^-1 -> 10x10x10 mesh)
SCF_SPACING = 0.2               # A^-1

# --- helper: write bands_post.in (bands.x input) ---
def write_bands_post_in(path, nnn, prefix, outdir):
    """Write bands.x input file using write_fortran_namelist.
    
    &BANDS namelist:
      prefix   -- must match pw.x prefix
      outdir   -- must match pw.x outdir
      filband  -- output band-structure file name
      lsym     -- .false. to skip band-representation file (saves time)
    """
    input_data = {
        'prefix': prefix,
        'outdir': outdir,
        'filband': f'{nnn}.bands.dat',
        'lsym': False,
    }
    with open(path, 'w') as f:
        write_fortran_namelist(f, input_data=input_data, binary='bands')


def main():
    os.makedirs(CALC_DIR, exist_ok=True)
    
    for n in range(N_STRUCTURES):
        nnn = f'{n:03d}'
        calc_path = os.path.join(CALC_DIR, nnn)
        os.makedirs(calc_path, exist_ok=True)
        
        # --- copy pseudopotential ---
        src_pseudo = os.path.join(TEMPLATE_DIR, PSEUDO_FILE)
        dst_pseudo = os.path.join(calc_path, PSEUDO_FILE)
        if not os.path.exists(dst_pseudo):
            shutil.copy2(src_pseudo, dst_pseudo)
        
        # --- read structure ---
        cif_path = os.path.join(STRUCTURES_DIR, f'structure_{nnn}.cif')
        atoms = read(cif_path, format='cif')
        
        # INVARIANT 1: nat/ntyp from the actual object
        nat = len(atoms)
        ntyp = len(set(atoms.get_chemical_symbols()))
        assert nat == 2, f'structure {nnn}: nat={nat}, expected 2'
        assert ntyp == 1, f'structure {nnn}: ntyp={ntyp}, expected 1'
        
        # --- k-path (INVARIANT 3) ---
        if n == 0:
            # Structure 000 is the unperturbed MP cell: use MP's k-path
            kp = load_kpath(os.path.join(STRUCTURES_DIR, 'kpath.json'))
        else:
            # Perturbed structures: use seekpath's standard path for this cell
            kp = standard_kpath(atoms)
        
        # The cell for pw.in and bands.in is kp_atoms(kp), NOT the CIF cell
        prim = kp_atoms(kp)
        
        # Save k-path JSON for this structure
        kp_save(kp, os.path.join(calc_path, 'kpath.json'))
        
        # --- SCF k-mesh (RULE: never typed by hand) ---
        kpts_scf = scf_kmesh(prim, spacing=SCF_SPACING)
        
        # --- bands k-points (INVARIANT 3) ---
        kpts_bands = kp_kpts(kp)
        
        # --- pseudopotential mapping for write_espresso_in ---
        # write_espresso_in expects {element: filename}
        pseudo_map = {'Si': PSEUDO_FILE}
        
        # --- input_data (FLAT dict; ASE routes each key to the correct namelist) ---
        # INVARIANT 6: do NOT hand-sort into &CONTROL / &SYSTEM / &ELECTRONS
        # pseudo_dir='.' tells pw.x to look for pseudopotentials in the current
        # directory (calculations/NNN/), where Si.upf has been copied.
        input_data = {
            'calculation': 'scf',
            'prefix': PREFIX,
            'outdir': OUTDIR,
            'pseudo_dir': '.',
            'ecutwfc': ECUTWFC_RY,
            'ecutrho': ECUTRHO_RY,
            'occupations': OCCUPATIONS,
            'conv_thr': CONV_THR,
            'nspin': 1,
        }
        
        # --- write pw.in (SCF) ---
        pw_in_path = os.path.join(calc_path, 'pw.in')
        write_espresso_in(
            pw_in_path,
            prim,
            input_data=input_data,
            pseudopotentials=pseudo_map,
            kpts=kpts_scf,
        )
        
        # --- write bands.in (non-self-consistent bands) ---
        bands_input_data = dict(input_data)
        bands_input_data['calculation'] = 'bands'
        bands_input_data['nbnd'] = NBND
        
        bands_in_path = os.path.join(calc_path, 'bands.in')
        write_espresso_in(
            bands_in_path,
            prim,
            input_data=bands_input_data,
            pseudopotentials=pseudo_map,
            kpts=kpts_bands,
        )
        
        # --- write bands_post.in (bands.x) ---
        bands_post_path = os.path.join(calc_path, 'bands_post.in')
        write_bands_post_in(bands_post_path, nnn, PREFIX, OUTDIR)
        
        print(f'[{nnn}] nat={nat} ntyp={ntyp} kpts_scf={kpts_scf} '
              f'n_kpts_bands={len(kpts_bands)} -> {calc_path}/')
    
    print(f'\nDone. Wrote inputs for {N_STRUCTURES} structures in {CALC_DIR}/')
    print(f'  ecutwfc={ECUTWFC_RY} Ry, ecutrho={ECUTRHO_RY} Ry, '
          f'conv_thr={CONV_THR} Ry, nbnd={NBND}, spacing={SCF_SPACING} A^-1')


if __name__ == '__main__':
    main()
