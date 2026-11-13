#!/usr/bin/env python
"""
Setup QE input files for all 51 TiO2 structures.

For each structure NNN (000-050):
  - Create calculations/NNN/
  - Copy pseudopotential files
  - Write kpath.json from chem_llm.kpath
  - Write pw.in (SCF) and bands.in (NSCF bands) using write_espresso_in
  - Write bands_post.in (bands.x input)

Key settings (from STEP 4 note):
  - ecutwfc = 92.0 Ry (max of Ti 84.0, O 92.0; converted from Ha: 46.0 Ha * 2 = 92.0 Ry)
  - ecutrho = 368.0 Ry (4 * ecutwfc)
  - nbnd = 48 (2x occupied bands; 48 electrons / 2 = 24 occupied, pad to 48)
  - occupations = 'fixed' (semiconductor, gap 1.77 eV)
  - conv_thr = 1.0e-8 Ry
  - nspin = 1 (non-magnetic)
  - prefix = 'tio2'
  - outdir = './tmp'
  - pseudo_dir = '.' (pseudopotentials in calculations/NNN/)
  - K-mesh: computed via scf_kmesh(prim, metal=False) from chem_llm.kpath
  - Bands kpts: kp_kpts(kp) from chem_llm.kpath
"""

import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# Pseudopotential files
PSEUDO_DIR = 'template'
PSEUDO_FILES = {
    'Ti': 'Ti.pbesol_upf',
    'O': 'O.pbesol_upf',
}

# Cutoffs (INVARIANT 5: converted from Ha to Ry, 1 Ha = 2 Ry)
# Ti: 42.0 Ha * 2 = 84.0 Ry
# O: 46.0 Ha * 2 = 92.0 Ry
# Use max: 92.0 Ry
ECUTWFC_RY = 92.0
ECUTRHO_RY = 4 * ECUTWFC_RY  # 368.0 Ry

# Electron count (INVARIANT 2): 2 Ti * 12 + 4 O * 6 = 48 electrons
# Occupied bands = 48/2 = 24 (spin-unpolarized)
# nbnd = 48 (2x occupied, safe for conduction bands)
NBND = 48

# Common settings
PREFIX = 'tio2'
OUTDIR = './tmp'
CONV_THR = 1.0e-8  # Ry

# Number of structures
N_STRUCTURES = 51

for nnn in range(N_STRUCTURES):
    calc_dir = f'calculations/{nnn:03d}'
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(f'{calc_dir}/tmp', exist_ok=True)
    
    # Copy pseudopotential files
    for elem, fname in PSEUDO_FILES.items():
        src = os.path.join(PSEUDO_DIR, fname)
        dst = os.path.join(calc_dir, fname)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    # Read the structure
    cif_path = f'structures/structure_{nnn:03d}.cif'
    atoms = ase.io.read(cif_path)
    
    # Get the k-path and primitive cell (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    
    # Save kpath.json
    kp_save(kp, f'{calc_dir}/kpath.json')
    
    # INVARIANT 1: nat and ntyp from the actual ASE Atoms object
    nat = len(prim)
    ntyp = len(set(prim.get_chemical_symbols()))
    assert nat == 6, f"Structure {nnn:03d}: expected 6 atoms, got {nat}"
    assert ntyp == 2, f"Structure {nnn:03d}: expected 2 types, got {ntyp}"
    
    # SCF k-mesh (computed, not hand-typed)
    kpts_scf = scf_kmesh(prim, metal=False)
    
    # Bands k-points from the k-path
    kpts_bands = kp_kpts(kp)
    
    # Pseudopotential mapping for write_espresso_in
    pseudos = {elem: fname for elem, fname in PSEUDO_FILES.items()}
    
    # SCF input (pw.in)
    input_data_scf = {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
        'nspin': 1,
    }
    
    with open(f'{calc_dir}/pw.in', 'w') as f:
        write_espresso_in(f, prim, input_data=input_data_scf,
                          pseudopotentials=pseudos, kpts=kpts_scf)
    
    # Bands input (bands.in) - NSCF band structure
    input_data_bands = {
        'calculation': 'bands',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
        'nspin': 1,
        'nbnd': NBND,
    }
    
    with open(f'{calc_dir}/bands.in', 'w') as f:
        write_espresso_in(f, prim, input_data=input_data_bands,
                          pseudopotentials=pseudos, kpts=kpts_bands)
    
    # Bands post-processing input (bands_post.in) for bands.x
    # &BANDS namelist: prefix, outdir, filband, lsym
    bands_post_content = f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{nnn:03d}.bands.dat'
    lsym = .false.
/
"""
    with open(f'{calc_dir}/bands_post.in', 'w') as f:
        f.write(bands_post_content)
    
    if nnn % 10 == 0:
        print(f"  Set up {calc_dir} (kpts_scf={kpts_scf}, n_kpts_bands={len(kpts_bands)})")

print(f"Done. Set up {N_STRUCTURES} calculation directories.")
