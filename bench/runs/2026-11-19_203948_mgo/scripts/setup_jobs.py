#!/usr/bin/env python
"""Set up QE input files for all 51 MgO structures.

For each structure NNN:
  - calculations/NNN/ directory
  - Pseudopotential files (Mg.upf, O.upf) copied in
  - kpath.json (MP's path for 000, standard_kpath for perturbed)
  - pw.in (SCF)
  - bands.in (NSCF bands)
  - bands_post.in (bands.x post-processing)

Key settings (from STEP 4 note):
  - ecutwfc = 35.0 Ry (from O's 17.5 Ha * 2; the larger of Mg=20 Ry, O=35 Ry)
  - ecutrho = 280.0 Ry (8 * ecutwfc, PAW requirement)
  - conv_thr = 1.0e-8 Ry
  - occupations = 'fixed' (MgO is a clear insulator, gap ~4.4 eV)
  - nspin = 1 (no magnetism)
  - nbnd = 20 (generous padding above 8 occupied bands)
  - prefix = 'mgo', outdir = './tmp', pseudo_dir = './'
  - No Hubbard U, no spin-orbit (scalar-relativistic pseudopotentials)
"""

import os
import shutil
import numpy as np
from ase.io import read, write
from ase.io.espresso import write_espresso_in
from chem_llm.dft.kpath import (
    standard_kpath, kp_atoms, kp_kpts, kp_save, load_kpath, scf_kmesh
)

# --- Constants ---
ECUTWFC_RY = 35.0   # Ry (from O's 17.5 Ha * 2; larger of Mg=20 Ry, O=35 Ry)
ECUTRHO_RY = 280.0  # Ry (8 * ecutwfc, PAW requirement)
CONV_THR = 1.0e-8   # Ry
NBND = 20           # generous padding above 8 occupied bands
PREFIX = 'mgo'
OUTDIR = './tmp'
PSUEDO_DIR = './'

# Pseudopotential files (relative to template/)
PSEUDO_FILES = {
    'Mg': 'template/Mg.upf',
    'O': 'template/O.upf',
}

# K-mesh spacing (converged in STEP 6: converge_kmesh returned spacing=0.25 A^-1)
K_SPACING = 0.25  # A^-1, from converge_kmesh result

# --- Main loop ---
for i in range(51):
    idx = f"{i:03d}"
    calc_dir = f"calculations/{idx}"
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotential files into calc_dir
    for sp, src in PSEUDO_FILES.items():
        dst = os.path.join(calc_dir, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    # Read structure
    cif_path = f"structures/structure_{idx}.cif"
    atoms = read(cif_path)
    
    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Structure {idx}: expected 2 atoms, got {nat}"
    assert ntyp == 2, f"Structure {idx}: expected 2 species, got {ntyp}"
    
    # Get k-path
    if i == 0:
        # Structure 000: use MP's k-path (belongs to the unperturbed cell)
        kp = load_kpath('structures/kpath.json')
    else:
        # Perturbed structures: use standard_kpath (seekpath)
        kp = standard_kpath(atoms)
    
    # Save kpath.json for this structure
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))
    
    # Get the primitive cell for the k-path (INVARIANT 3)
    prim = kp_atoms(kp)
    
    # Compute SCF k-mesh from the primitive cell
    kpts = scf_kmesh(prim, spacing=K_SPACING)
    
    # Pseudopotential filenames (relative to calc_dir, where pw.x runs)
    pseudos = {
        'Mg': 'Mg.upf',
        'O': 'O.upf',
    }
    
    # --- Write pw.in (SCF) ---
    input_data_scf = {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': PSUEDO_DIR,
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'conv_thr': CONV_THR,
        'occupations': 'fixed',
        'nspin': 1,
    }
    
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=input_data_scf,
        pseudopotentials=pseudos,
        kpts=kpts,
    )
    
    # --- Write bands.in (NSCF bands) ---
    input_data_bands = {
        'calculation': 'bands',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': PSUEDO_DIR,
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'conv_thr': CONV_THR,
        'occupations': 'fixed',
        'nspin': 1,
        'nbnd': NBND,
    }
    
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=input_data_bands,
        pseudopotentials=pseudos,
        kpts=kp_kpts(kp),
    )
    
    # --- Write bands_post.in (bands.x) ---
    # bands.x input is a simple namelist, written manually
    bands_post_content = f"""&BANDS
  prefix = '{PREFIX}'
  outdir = '{OUTDIR}'
  filband = '{idx}.bands.dat'
  lsym = .false.
  spin_component = 1
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)
    
    print(f"Wrote {calc_dir}/: pw.in, bands.in, bands_post.in, kpath.json")
    print(f"  nat={nat}, ntyp={ntyp}, kpts={kpts}, nbnd={NBND}")

print("\nDone. All 51 calculation directories set up.")
