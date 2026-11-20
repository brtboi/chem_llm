#!/usr/bin/env python3
"""
STEP 5: Generate QE input files for all 51 TiO2 structures.

For each structure NNN (000-050), creates calculations/NNN/ containing:
  - Ti.upf, O.upf (pseudopotentials copied from template/)
  - kpath.json (MP's path for 000, seekpath for perturbed)
  - pw.in (SCF input)
  - bands.in (NSCF bands input)
  - bands_post.in (bands.x post-processing input)

Key settings (from STEP 4 note):
  - ecutwfc = 35.0 Ry (from O's 17.5 Ha * 2, INVARIANT 5)
  - ecutrho = 280.0 Ry (8x ecutwfc for PAW)
  - occupations = 'fixed' (TiO2 is an insulator, gap ~1.78 eV)
  - conv_thr = 1.0e-8 Ry
  - nbnd = 40 (occupied = 48/2 = 24, padded)
  - No Hubbard U, no spin polarization, no spin-orbit
  - SCF k-mesh: scf_kmesh(prim, spacing=0.25) -- converged by converge_kmesh
"""

import os
import shutil
import numpy as np
from ase.io import read, write
from ase.io.espresso import write_espresso_in
from chem_llm.dft.kpath import (
    standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh, load_kpath
)

# --- Configuration ---
N_STRUCTURES = 51
TEMPLATE_DIR = 'template'
STRUCTURES_DIR = 'structures'
CALC_DIR = 'calculations'

# Cutoffs (INVARIANT 5: convert Ha to Ry, 1 Ha = 2 Ry)
# O: 17.5 Ha = 35.0 Ry (limiting element)
# Ti: 10.0 Ha = 20.0 Ry
ECUTWFC_RY = 35.0  # Ry
ECUTRHO_RY = 8.0 * ECUTWFC_RY  # 280.0 Ry (PAW needs 8x)

# SCF settings
PREFIX = 'tio2'
OUTDIR = './tmp'
CONV_THR = 1.0e-8  # Ry

# Bands settings
NBND = 40  # occupied = 48/2 = 24, padded to 40

# K-mesh spacing (converged by converge_kmesh on calculations/000)
KSPACING = 0.25  # A^-1, from converge_kmesh result

# Pseudopotential files
PSEUDO_FILES = {
    'Ti': 'Ti.upf',
    'O': 'O.upf',
}


def setup_structure(nnn: int, kspacing: float):
    """Set up QE input files for structure NNN."""
    struct_path = f'{STRUCTURES_DIR}/structure_{nnn:03d}.cif'
    calc_path = f'{CALC_DIR}/{nnn:03d}'
    
    # Create directory
    os.makedirs(calc_path, exist_ok=True)
    
    # Copy pseudopotentials
    for elem, fname in PSEUDO_FILES.items():
        src = os.path.join(TEMPLATE_DIR, fname)
        dst = os.path.join(calc_path, fname)
        shutil.copy2(src, dst)
    
    # Read structure
    atoms = read(struct_path)
    
    # Get k-path
    if nnn == 0:
        # Use MP's k-path (from structures/kpath.json)
        kp = load_kpath(f'{STRUCTURES_DIR}/kpath.json')
    else:
        # Use seekpath's standard k-path for the perturbed structure
        kp = standard_kpath(atoms)
    
    # Get the cell for the calculation (kp_atoms returns the primitive cell)
    prim = kp_atoms(kp)
    
    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(prim)
    ntyp = len(set(prim.get_chemical_symbols()))
    assert nat == 6, f"Structure {nnn:03d}: expected 6 atoms, got {nat}"
    assert ntyp == 2, f"Structure {nnn:03d}: expected 2 species, got {ntyp}"
    
    # Save k-path JSON
    kp_save(kp, os.path.join(calc_path, 'kpath.json'))
    
    # SCF k-mesh (Monkhorst-Pack, even spacing)
    kpts_scf = scf_kmesh(prim, spacing=kspacing)
    
    # Bands k-points (from the k-path)
    kpts_bands = kp_kpts(kp)
    
    # Pseudopotential mapping
    pseudos = {
        'Ti': 'Ti.upf',
        'O': 'O.upf',
    }
    
    # --- Write pw.in (SCF) ---
    input_data_scf = {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
        'pseudo_dir': '.',
    }
    
    pw_in_path = os.path.join(calc_path, 'pw.in')
    write_espresso_in(pw_in_path, prim, input_data=input_data_scf,
                      pseudopotentials=pseudos, kpts=kpts_scf)
    
    # --- Write bands.in (NSCF bands) ---
    input_data_bands = {
        'calculation': 'bands',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
        'nbnd': NBND,
        'pseudo_dir': '.',
    }
    
    bands_in_path = os.path.join(calc_path, 'bands.in')
    write_espresso_in(bands_in_path, prim, input_data=input_data_bands,
                      pseudopotentials=pseudos, kpts=kpts_bands)
    
    # --- Write bands_post.in (bands.x) ---
    bands_post_path = os.path.join(calc_path, 'bands_post.in')
    with open(bands_post_path, 'w') as f:
        f.write('&BANDS\n')
        f.write(f"  prefix = '{PREFIX}'\n")
        f.write(f"  outdir = '{OUTDIR}'\n")
        f.write(f"  filband = '{nnn:03d}.bands.dat'\n")
        f.write('  lsym = .false.\n')
        f.write('/\n')
    
    print(f"Structure {nnn:03d}: {nat} atoms, {ntyp} species, "
          f"kpts_scf={kpts_scf}, kpts_bands={len(kpts_bands)}")


def main():
    # Create top-level calculations directory
    os.makedirs(CALC_DIR, exist_ok=True)
    
    for nnn in range(N_STRUCTURES):
        setup_structure(nnn, KSPACING)
    
    print(f"\nDone: set up {N_STRUCTURES} calculation directories in {CALC_DIR}/")


if __name__ == '__main__':
    main()
