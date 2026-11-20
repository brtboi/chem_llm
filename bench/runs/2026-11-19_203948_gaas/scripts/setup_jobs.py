#!/usr/bin/env python3
"""Generate QE input files for all 51 GaAs structures.

For each structure NNN (000-050):
  - calculations/NNN/ directory
  - Pseudopotential files (Ga.upf, As.upf) copied from template/
  - kpath.json (MP's path for 000, seekpath for perturbed)
  - pw.in (SCF)
  - bands.in (non-self-consistent bands)
  - bands_post.in (bands.x post-processing)

Key settings (from STEP 4 note):
  - ecutwfc = 24.0 Ry (12.0 Ha * 2, INVARIANT 5)
  - ecutrho = 192.0 Ry (8 * ecutwfc, PAW)
  - occupations = 'fixed' (semiconductor)
  - conv_thr = 1.0e-8 Ry
  - nspin = 1 (non-magnetic)
  - No Hubbard U, no spin-orbit
  - nbnd = 20 (9 occupied bands, padded)
  - pseudo_dir = './' (pseudopotentials are in the same directory as pw.in)
  - SCF k-mesh spacing = 0.2 (converged by converge_kmesh in STEP 6)
"""

import os
import shutil
import numpy as np
from ase.io import read, write
from ase.io.espresso import write_espresso_in

# Import kpath utilities
from chem_llm.dft.kpath import (
    standard_kpath, kp_atoms, kp_kpts, kp_save, load_kpath, scf_kmesh
)

# Pseudopotential files
PSEUDO_FILES = {
    'Ga': 'template/Ga.upf',
    'As': 'template/As.upf',
}

# QE settings
PREFIX = 'gaas'
OUTDIR = './tmp'
ECUTWFC_RY = 24.0   # 12.0 Ha * 2 (INVARIANT 5)
ECUTRHO_RY = 192.0  # 8 * ecutwfc (PAW)
CONV_THR = 1.0e-8   # Ry
NBND = 20           # 9 occupied bands, padded
PSEUDO_DIR = './'   # Pseudopotentials are in the same directory as pw.in

# SCF k-mesh spacing (converged by converge_kmesh in STEP 6)
SCF_SPACING = 0.2


def setup_structure(nnn: int):
    """Set up QE input files for structure NNN."""
    struct_path = f'structures/structure_{nnn:03d}.cif'
    calc_dir = f'calculations/{nnn:03d}'
    os.makedirs(calc_dir, exist_ok=True)

    # Read structure
    atoms = read(struct_path)
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    print(f"Structure {nnn:03d}: {nat} atoms, {ntyp} types")

    # Copy pseudopotential files
    for element, src in PSEUDO_FILES.items():
        dst = os.path.join(calc_dir, os.path.basename(src))
        shutil.copy2(src, dst)

    # Get k-path
    if nnn == 0:
        # Use MP's k-path (belongs to the unperturbed cell)
        kp = load_kpath('structures/kpath.json')
    else:
        # Perturbed structure: use seekpath's standard path
        kp = standard_kpath(atoms)

    # Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))

    # Get the primitive cell for the k-path
    prim = kp_atoms(kp)
    print(f"  k-path cell: {len(prim)} atoms")

    # Pseudopotential filenames (relative to calc_dir)
    pseudos = {
        'Ga': 'Ga.upf',
        'As': 'As.upf',
    }

    # SCF k-mesh
    kpts_scf = scf_kmesh(prim, spacing=SCF_SPACING)
    print(f"  SCF k-mesh: {kpts_scf}")

    # Bands k-points
    kpts_bands = kp_kpts(kp)
    print(f"  Bands k-points: {len(kpts_bands)}")

    # SCF input data (flat dict, ASE routes to correct namelists)
    scf_input_data = {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': PSEUDO_DIR,
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
        'nspin': 1,
    }

    # Write pw.in (SCF)
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=scf_input_data,
        pseudopotentials=pseudos,
        kpts=kpts_scf,
    )

    # Bands input data
    bands_input_data = {
        'calculation': 'bands',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': PSEUDO_DIR,
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'nbnd': NBND,
        'nspin': 1,
    }

    # Write bands.in
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=bands_input_data,
        pseudopotentials=pseudos,
        kpts=kpts_bands,
    )

    # bands.x post-processing input
    # Write manually since write_espresso_in is for pw.x
    bands_post_content = f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{nnn:03d}.bands.dat'
    lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)

    print(f"  Wrote pw.in, bands.in, bands_post.in to {calc_dir}/")


def main():
    print("Setting up QE input files for all 51 structures...")
    for nnn in range(51):
        setup_structure(nnn)
    print("\nDone. All 51 calculation directories created.")


if __name__ == '__main__':
    main()
