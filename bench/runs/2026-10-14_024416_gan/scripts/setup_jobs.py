#!/usr/bin/env python3
"""
setup_jobs.py -- Generate QE input files for wurtzite GaN band-structure calculations.

For each structure NNN (000..050), creates calculations/NNN/ containing:
  - Pseudopotential files (copied from template/)
  - kpath.json (from chem_llm.kpath)
  - pw.in (SCF input, written with ase.io.espresso.write_espresso_in)
  - bands.in (NSCF bands input, written with ase.io.espresso.write_espresso_in)
  - bands_post.in (bands.x post-processing input, hand-written namelist)

QE settings (from STEP 4 note):
  ecutwfc = 114.0 Ry  (2 * 57.0 Ha, Ga is limiting)
  ecutrho = 456.0 Ry  (4 * ecutwfc)
  occupations = 'fixed' (GaN is a semiconductor, gap ~1.74 eV)
  conv_thr = 1.0e-8 Ry
  nspin = 1 (non-magnetic)
  nbnd = 40 (pad above 26 occupied bands)
  prefix = 'gan', outdir = 'tmp', pseudo_dir = '.'
"""

import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# --- Pseudopotential files ---
PP_FILES = {
    'Ga': 'template/Ga.pbesol-rrkjus_psl.1.0.0.UPF',
    'N':  'template/N.pbesol-rrkjus_psl.1.0.0.UPF',
}

# --- QE parameters ---
ECUTWFC_RY = 114.0   # Ry (2 * 57.0 Ha)
ECUTRHO_RY = 456.0   # Ry (4 * ecutwfc)
CONV_THR   = 1.0e-8  # Ry
NBND       = 40      # pad above 26 occupied bands
PREFIX     = 'gan'
OUTDIR     = 'tmp'

N_STRUCTURES = 51  # 000..050

# --- Main loop ---
for i in range(N_STRUCTURES):
    nnn = f"{i:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)

    # Copy pseudopotential files into calc_dir (pseudo_dir='.')
    for sym, pp_path in PP_FILES.items():
        dst = os.path.join(calc_dir, os.path.basename(pp_path))
        if not os.path.exists(dst):
            shutil.copy2(pp_path, dst)

    # Read structure
    atoms = ase.io.read(f"structures/structure_{nnn}.cif")
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    print(f"Structure {nnn}: nat={nat}, ntyp={ntyp}")

    # INVARIANT 1: nat/ntyp from actual Atoms object
    assert nat == 4, f"Structure {nnn}: expected nat=4, got {nat}"
    assert ntyp == 2, f"Structure {nnn}: expected ntyp=2, got {ntyp}"

    # --- K-path (INVARIANT 3) ---
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))

    # Pseudopotential filenames (relative to calc_dir, pseudo_dir='.')
    pseudopotentials = {
        'Ga': os.path.basename(PP_FILES['Ga']),
        'N':  os.path.basename(PP_FILES['N']),
    }

    # --- SCF k-mesh (DECISION RULE: use scf_kmesh, not hand-typed) ---
    scf_kpts = scf_kmesh(prim, metal=False)
    print(f"  SCF k-mesh: {scf_kpts}")

    # --- SCF input (pw.in) ---
    scf_input_data = {
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
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=scf_input_data,
        pseudopotentials=pseudopotentials,
        kpts=scf_kpts,
    )

    # --- Bands input (bands.in) ---
    bands_input_data = {
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
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=bands_input_data,
        pseudopotentials=pseudopotentials,
        kpts=kp_kpts(kp),  # INVARIANT 3: k-points from kpath
    )

    # --- Bands post-processing input (bands_post.in) ---
    # bands.x input: &BANDS namelist
    # lsym=.false. to skip band-representation file (not needed)
    bands_post = f"""&BANDS
  prefix = '{PREFIX}'
  outdir = '{OUTDIR}'
  filband = '{nnn}.bands.dat'
  lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post)

    print(f"  Wrote {calc_dir}/pw.in, bands.in, bands_post.in, kpath.json")

print(f"\nDone. Set up {N_STRUCTURES} calculation directories.")
