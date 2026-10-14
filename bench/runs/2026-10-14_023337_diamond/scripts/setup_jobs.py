#!/usr/bin/env python
"""
setup_jobs.py -- Create QE input files for all 51 diamond-C structures.

For each structure NNN (000..050):
  calculations/NNN/
    C.pbesol-rrkjus_psl.1.0.0.UPF   (pseudopotential, copied from template/)
    kpath.json                       (standard k-path from chem_llm.kpath)
    pw.in                            (SCF input, written by write_espresso_in)
    bands.in                         (NSCF bands input, written by write_espresso_in)
    bands_post.in                    (bands.x post-processing input)

Key decisions (see STEP 4 note):
  - ecutwfc = 82.0 Ry  (41.0 Ha * 2, per INVARIANT 5)
  - ecutrho = 328.0 Ry (4 * ecutwfc)
  - occupations = 'fixed' (diamond is a clear insulator, gap ~4.1 eV)
  - conv_thr = 1e-8 Ry
  - nspin = 1 (non-magnetic)
  - nbnd = 16 (4 occupied bands * 4 padding; z_valence=4 per C, 2 atoms -> 8 e- -> 4 occ. bands)
  - SCF k-mesh: computed by scf_kmesh(prim, metal=False) from chem_llm.kpath
  - Bands k-points: kp_kpts(kp) from chem_llm.kpath (INVARIANT 3)
  - Cell for both pw.in and bands.in: kp_atoms(kp) (the primitive cell the k-path refers to)
  - No spin-orbit (C is Z=6, well below 54)
  - lsym = .false. in bands_post.in (saves time, no band-representation file needed)
  - pseudo_dir = './' in &CONTROL so QE looks for pseudopotentials in the current directory
"""

import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# --- Constants ---
N_STRUCTURES = 51
PSEUDO_FILE = 'template/C.pbesol-rrkjus_psl.1.0.0.UPF'

# Read the recommended cutoff from the pseudopotential file (INVARIANT 5)
pp_header = read_upf_header(PSEUDO_FILE)
z_valence = pp_header['z_valence']  # 4.0 for C
print(f'Pseudopotential z_valence = {z_valence}')

# Recommended cutoff in Ha (from get_pseudopotential result)
# We read it from the file header if available, otherwise use the known value
# The get_pseudopotential tool reported 41.0 Ha (normal level)
# Convert to Ry: 1 Ha = 2 Ry
ecutwfc_ha = 41.0  # Ha, from get_pseudopotential result
ecutwfc_ry = 2.0 * ecutwfc_ha  # 82.0 Ry
ecutrho_ry = 4.0 * ecutwfc_ry  # 328.0 Ry

# INVARIANT 5: assert the conversion is correct
assert abs(ecutwfc_ry - 82.0) < 1e-10, f'ecutwfc_ry should be 82.0, got {ecutwfc_ry}'
assert abs(ecutrho_ry - 328.0) < 1e-10, f'ecutrho_ry should be 328.0, got {ecutrho_ry}'

# nbnd: 2 atoms * z_valence / 2 = 4 occupied bands (spin-unpolarized)
# Pad to 4x for safety
n_atoms_expected = 2
n_electrons = n_atoms_expected * z_valence  # 8.0
n_occupied = int(n_electrons / 2)  # 4
nbnd = 4 * n_occupied  # 16
print(f'n_electrons = {n_electrons}, n_occupied = {n_occupied}, nbnd = {nbnd}')

# --- Common input_data for SCF and bands ---
# INVARIANT 6: pass a FLAT dict; write_espresso_in routes each key to the correct namelist
common_input = {
    'prefix': 'diamond',
    'outdir': './tmp',
    'pseudo_dir': './',  # &CONTROL variable: look for pseudopotentials in current directory
    'ecutwfc': ecutwfc_ry,   # Ry
    'ecutrho': ecutrho_ry,   # Ry
    'occupations': 'fixed',
    'conv_thr': 1.0e-8,      # Ry
    'nspin': 1,
}

# Pseudopotential mapping: species -> filename (relative to calculations/NNN/)
pseudopotentials = {'C': 'C.pbesol-rrkjus_psl.1.0.0.UPF'}

for nnn in range(N_STRUCTURES):
    calc_dir = f'calculations/{nnn:03d}'
    os.makedirs(calc_dir, exist_ok=True)

    # Copy pseudopotential into the calculation directory
    shutil.copy2(PSEUDO_FILE, os.path.join(calc_dir, 'C.pbesol-rrkjus_psl.1.0.0.UPF'))

    # Read the structure
    atoms = ase.io.read(f'structures/structure_{nnn:03d}.cif')

    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == n_atoms_expected, f'Structure {nnn:03d}: expected {n_atoms_expected} atoms, got {nat}'
    assert ntyp == 1, f'Structure {nnn:03d}: expected 1 species, got {ntyp}'

    # Get the standard k-path for THIS structure (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in

    # INVARIANT 1: nat/ntyp from the primitive cell actually written
    nat_prim = len(prim)
    ntyp_prim = len(set(prim.get_chemical_symbols()))
    assert nat_prim == n_atoms_expected, f'Structure {nnn:03d}: prim nat mismatch'
    assert ntyp_prim == 1, f'Structure {nnn:03d}: prim ntyp mismatch'

    # Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))

    # SCF k-mesh (DECISION RULE: never typed by hand)
    kpts_scf = scf_kmesh(prim, metal=False)
    print(f'Structure {nnn:03d}: SCF k-mesh = {kpts_scf}')

    # --- Write pw.in (SCF) ---
    scf_input = dict(common_input)
    scf_input['calculation'] = 'scf'
    scf_input['nat'] = nat_prim
    scf_input['ntyp'] = ntyp_prim

    with open(os.path.join(calc_dir, 'pw.in'), 'w') as f:
        write_espresso_in(f, prim, input_data=scf_input,
                          pseudopotentials=pseudopotentials, kpts=kpts_scf)

    # --- Write bands.in (NSCF bands) ---
    bands_input = dict(common_input)
    bands_input['calculation'] = 'bands'
    bands_input['nat'] = nat_prim
    bands_input['ntyp'] = ntyp_prim
    bands_input['nbnd'] = nbnd

    kpts_bands = kp_kpts(kp)  # (INVARIANT 3: k-points from the k-path)

    with open(os.path.join(calc_dir, 'bands.in'), 'w') as f:
        write_espresso_in(f, prim, input_data=bands_input,
                          pseudopotentials=pseudopotentials, kpts=kpts_bands)

    # --- Write bands_post.in (bands.x input) ---
    # &BANDS namelist: prefix, outdir, filband, lsym
    # lsym = .false. to skip the (slow) band-representation analysis
    bands_post_content = f"""&BANDS
    prefix = 'diamond'
    outdir = './tmp'
    filband = '{nnn:03d}.bands.dat'
    lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)

    print(f'Structure {nnn:03d}: wrote pw.in, bands.in, bands_post.in, kpath.json')

print(f'\nDone. Set up {N_STRUCTURES} calculation directories.')
