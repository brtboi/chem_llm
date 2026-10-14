#!/usr/bin/env python3
"""
setup_jobs.py -- Generate QE input files for all 51 GaAs structures.

For each structure NNN (000-050):
  calculations/NNN/
    Ga.pbesol-srn-rrkjus_psl.1.0.0.UPF   (copied from template/)
    As.pbesol-srn-rrkjus_psl.1.0.0.UPF   (copied from template/)
    kpath.json                            (from chem_llm.kpath)
    pw.in                                 (SCF input)
    bands.in                              (NSCF bands input)
    bands_post.in                         (bands.x post-processing input)

Key decisions (see STEP 4 note):
  - ecutwfc = 104 Ry  (min of Ga 57 Ha = 114 Ry, As 52 Ha = 104 Ry; INVARIANT 5: 1 Ha = 2 Ry)
  - ecutrho = 416 Ry  (4 * ecutwfc)
  - occupations = 'fixed' (GaAs is a semiconductor, not a metal)
  - conv_thr = 1.0d-8
  - nbnd = 40 (44 total electrons / 2 = 22 occupied bands, padded well above)
  - SCF k-mesh: from chem_llm.kpath.scf_kmesh(prim, metal=False)
  - Bands k-path: from chem_llm.kpath.kp_kpts(kp) (INVARIANT 3)
  - Pseudopotentials: scalar-relativistic (Ga Z=31, As Z=33, both < 54)
  - No spin-orbit (nspin=1, noncolin not set)
"""

import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# --- Pseudopotential files ---
GA_PSP = 'template/Ga.pbesol-srn-rrkjus_psl.1.0.0.UPF'
AS_PSP = 'template/As.pbesol-srn-rrkjus_psl.1.0.0.UPF'

# --- Cutoffs (converted from Ha to Ry per INVARIANT 5) ---
# Ga: 57 Ha = 114 Ry, As: 52 Ha = 104 Ry
# Use the minimum to ensure both are converged
ECUTWFC_RY = 104.0
ECUTRHO_RY = 4.0 * ECUTWFC_RY  # 416 Ry

# --- Electron count (from z_valence of pseudopotentials) ---
# Ga: z_valence = 21, As: z_valence = 23
# Total electrons = 21 + 23 = 44
# Occupied bands (spin-unpolarized) = 44 / 2 = 22
# nbnd padded well above occupied count
NBND = 40

# --- Common input_data for SCF and bands ---
# INVARIANT 6: pass a FLAT dict; ASE routes each key to the correct namelist
COMMON_INPUT = {
    'prefix': 'gaas',
    'outdir': './tmp',
    'pseudo_dir': './',
    'ecutwfc': ECUTWFC_RY,
    'ecutrho': ECUTRHO_RY,
    'occupations': 'fixed',
    'conv_thr': 1.0e-8,
}

# Pseudopotential mapping for write_espresso_in
PSEUDOS = {
    'Ga': 'Ga.pbesol-srn-rrkjus_psl.1.0.0.UPF',
    'As': 'As.pbesol-srn-rrkjus_psl.1.0.0.UPF',
}

N_STRUCTURES = 51  # 000 through 050

for nnn in range(N_STRUCTURES):
    label = f'{nnn:03d}'
    calc_dir = f'calculations/{label}'
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(f'{calc_dir}/tmp', exist_ok=True)

    # Copy pseudopotentials into the calculation directory
    for psp in [GA_PSP, AS_PSP]:
        shutil.copy2(psp, calc_dir)

    # Read the structure
    cif_path = f'structures/structure_{label}.cif'
    atoms = ase.io.read(cif_path)

    # INVARIANT 1: nat and ntyp computed from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f'structure_{label}: expected 2 atoms, got {nat}'
    assert ntyp == 2, f'structure_{label}: expected 2 types, got {ntyp}'

    # Get the standard k-path for THIS structure (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in

    # Save kpath.json (holds labels, ticks, segments)
    kp_save(kp, f'{calc_dir}/kpath.json')

    # --- SCF k-mesh (Monkhorst-Pack, not a hand-typed value) ---
    # metal=False: GaAs is a semiconductor
    scf_kpts = scf_kmesh(prim, metal=False)

    # --- Write pw.in (SCF) ---
    scf_input = dict(COMMON_INPUT)
    scf_input['calculation'] = 'scf'
    write_espresso_in(
        f'{calc_dir}/pw.in',
        prim,
        input_data=scf_input,
        pseudopotentials=PSEUDOS,
        kpts=scf_kpts,
    )

    # --- Write bands.in (NSCF bands) ---
    bands_input = dict(COMMON_INPUT)
    bands_input['calculation'] = 'bands'
    bands_input['nbnd'] = NBND
    write_espresso_in(
        f'{calc_dir}/bands.in',
        prim,
        input_data=bands_input,
        pseudopotentials=PSEUDOS,
        kpts=kp_kpts(kp),  # INVARIANT 3: k-points from the k-path
    )

    # --- Write bands_post.in (bands.x input) ---
    # bands.x namelist &BANDS: prefix, outdir, filband, lsym
    # lsym=.false. to skip band-representation file (saves time)
    bands_post_content = f"""&BANDS
    prefix = 'gaas'
    outdir = './tmp'
    filband = '{label}.bands.dat'
    lsym = .false.
/
"""
    with open(f'{calc_dir}/bands_post.in', 'w') as f:
        f.write(bands_post_content)

    if nnn <= 2 or nnn % 10 == 0:
        print(f'Wrote {calc_dir}/ (nat={nat}, ntyp={ntyp}, scf_kpts={scf_kpts})')

print(f'\nDone. Set up {N_STRUCTURES} calculation directories (000-050).')
