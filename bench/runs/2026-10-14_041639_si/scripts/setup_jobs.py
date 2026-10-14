#!/usr/bin/env python3
"""
setup_jobs.py -- For each structure NNN (000..050), create calculations/NNN/ with:
  - pseudopotential files (copied from template/)
  - kpath.json (from chem_llm.kpath)
  - pw.in (SCF input, written with write_espresso_in)
  - bands.in (NSCF bands input, written with write_espresso_in)
  - bands_post.in (bands.x post-processing input)

All pw.x settings decided in STEP 4:
  - ecutwfc = 36.0 Ry (18.0 Ha * 2, per INVARIANT 5)
  - ecutrho = 144.0 Ry (4 * ecutwfc)
  - occupations = 'fixed' (Si is a semiconductor, gap ~0.61 eV)
  - conv_thr = 1e-8 Ry
  - nspin = 1 (non-magnetic)
  - nbnd = 16 (8 electrons / 2 = 4 occupied bands, padded to 16)
  - kpts for SCF: scf_kmesh(prim, metal=False) from chem_llm.kpath
  - kpts for bands: kp_kpts(kp) from chem_llm.kpath (INVARIANT 3)
"""

import os
import shutil
import numpy as np
from ase.io import read, write
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# --- Constants ---
ECUTWFC_RY = 36.0   # 18.0 Ha * 2 (INVARIANT 5: 1 Ha = 2 Ry)
ECUTRHO_RY = 144.0  # 4 * ecutwfc
CONV_THR = 1e-8     # Ry
NBND = 16           # 8 electrons / 2 = 4 occupied bands, padded to 16
PREFIX = 'si'
OUTDIR = 'tmp'

# Pseudopotential filename (from template/)
PP_FILE = 'Si.pbesol-rrkjus_psl.1.0.0.UPF'

# --- Read base structure and reduce to primitive cell ---
base = read('structures/structure_000.cif')
kp = standard_kpath(base)
prim = kp_atoms(kp)

print(f"Primitive cell: {len(prim)} atoms, space group {kp['spacegroup']}")
assert len(prim) == 2, "Diamond Si primitive cell must have 2 atoms"

# --- SCF k-point mesh ---
scf_kpts = scf_kmesh(prim, metal=False)
print(f"SCF k-point mesh: {scf_kpts}")

# --- Loop over all 51 structures ---
for nnn in range(51):
    calc_dir = f'calculations/{nnn:03d}'
    os.makedirs(calc_dir, exist_ok=True)

    # Copy pseudopotential into calc_dir (pseudo_dir='.' resolves from here)
    pp_src = f'template/{PP_FILE}'
    pp_dst = f'{calc_dir}/{PP_FILE}'
    if not os.path.exists(pp_dst):
        shutil.copy2(pp_src, pp_dst)

    # Read the structure for this NNN
    atoms = read(f'structures/structure_{nnn:03d}.cif')

    # INVARIANT 1: nat/ntyp from actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Structure {nnn:03d}: expected 2 atoms, got {nat}"
    assert ntyp == 1, f"Structure {nnn:03d}: expected 1 species, got {ntyp}"

    # Get k-path for THIS structure (perturbed structures have lower symmetry)
    kp_nnn = standard_kpath(atoms)
    prim_nnn = kp_atoms(kp_nnn)
    kp_save(kp_nnn, f'{calc_dir}/kpath.json')

    # --- Write pw.in (SCF) ---
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
    write_espresso_in(
        f'{calc_dir}/pw.in',
        prim_nnn,
        input_data=input_data_scf,
        pseudopotentials={'Si': PP_FILE},
        kpts=scf_kpts,
    )

    # --- Write bands.in (NSCF bands) ---
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
    write_espresso_in(
        f'{calc_dir}/bands.in',
        prim_nnn,
        input_data=input_data_bands,
        pseudopotentials={'Si': PP_FILE},
        kpts=kp_kpts(kp_nnn),
    )

    # --- Write bands_post.in (bands.x input) ---
    # bands.x reads from the outdir written by pw.x; filband is the output file
    bands_post_content = f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{nnn:03d}.bands.dat'
    lsym = .false.
/
"""
    with open(f'{calc_dir}/bands_post.in', 'w') as f:
        f.write(bands_post_content)

    print(f"Wrote {calc_dir}/ (pw.in, bands.in, bands_post.in, kpath.json, {PP_FILE})")

print("Done.")
