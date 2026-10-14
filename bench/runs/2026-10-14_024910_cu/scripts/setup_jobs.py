#!/usr/bin/env python3
"""
setup_jobs.py -- Generate QE input files for all 51 Cu structures.

For each structure NNN (000-050):
  calculations/NNN/
    Cu.pbesol-sr-rrkjus_psl.1.0.0.UPF  (pseudopotential, copied from template/)
    kpath.json                          (from chem_llm.kpath)
    pw.in                               (SCF input, written by write_espresso_in)
    bands.in                            (bands input, written by write_espresso_in)
    bands_post.in                       (bands.x input, hand-written)

Key settings (from STEP 4 note):
  - ecutwfc = 104.0 Ry  (52.0 Ha * 2.0 Ha/Ry, INVARIANT 5)
  - ecutrho = 416.0 Ry  (4 * ecutwfc)
  - occupations = 'smearing', smearing = 'mv', degauss = 0.01 Ry (Cu is a metal)
  - conv_thr = 1e-8
  - nbnd = 20 (19 electrons -> 9.5 occupied bands, padded)
  - SCF k-mesh: scf_kmesh(prim, metal=True)
  - Bands k-points: kp_kpts(kp) from chem_llm.kpath (INVARIANT 3)
  - bands_post.in: prefix, outdir, filband, lsym=.false.
"""

import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_STRUCTURES = 51
PSEUDO_FILE = 'template/Cu.pbesol-sr-rrkjus_psl.1.0.0.UPF'

# Read the pseudopotential header to get z_valence (INVARIANT 2)
pp_header = read_upf_header(PSEUDO_FILE)
z_valence = pp_header['z_valence']
print(f"Pseudopotential z_valence: {z_valence}")

# Recommended cutoff in Hartree (from get_pseudopotential result)
ecutwfc_ha = 52.0  # Ha, from get_pseudopotential recommended_ecutwfc
# INVARIANT 5: convert Ha -> Ry (1 Ha = 2 Ry)
ecutwfc_ry = 2.0 * ecutwfc_ha
ecutrho_ry = 4.0 * ecutwfc_ry
print(f"ecutwfc = {ecutwfc_ry} Ry, ecutrho = {ecutrho_ry} Ry")

# nbnd: 1 atom * z_valence electrons -> occupied bands = z_valence / 2
# Pad nbnd above that
n_electrons = 1 * z_valence  # 1 atom per primitive cell
n_occupied = n_electrons / 2.0  # spin-unpolarized
nbnd = int(np.ceil(n_occupied)) + 10  # pad by 10
print(f"n_electrons = {n_electrons}, n_occupied = {n_occupied}, nbnd = {nbnd}")

# ---------------------------------------------------------------------------
# Loop over structures
# ---------------------------------------------------------------------------
for nnn in range(N_STRUCTURES):
    nnn_str = f"{nnn:03d}"
    calc_dir = f"calculations/{nnn_str}"
    os.makedirs(calc_dir, exist_ok=True)

    # Copy pseudopotential into the calculation directory
    shutil.copy2(PSEUDO_FILE, calc_dir)

    # Read the structure
    atoms = ase.io.read(f"structures/structure_{nnn_str}.cif")

    # INVARIANT 1: compute nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 1, f"Expected 1 atom, got {nat}"
    assert ntyp == 1, f"Expected 1 type, got {ntyp}"

    # Get the standard k-path and the primitive cell it refers to (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))

    # SCF k-mesh (metal=True for Cu)
    kpts_scf = scf_kmesh(prim, metal=True)
    print(f"Structure {nnn_str}: SCF k-mesh = {kpts_scf}")

    # Bands k-points from the k-path
    kpts_bands = kp_kpts(kp)
    print(f"Structure {nnn_str}: bands k-points = {len(kpts_bands)}")

    # Pseudopotential filename (relative to calc_dir)
    pseudo_name = os.path.basename(PSEUDO_FILE)
    pseudos = {'Cu': pseudo_name}

    # -----------------------------------------------------------------------
    # SCF input (pw.in)
    # -----------------------------------------------------------------------
    input_data_scf = {
        'calculation': 'scf',
        'prefix': f'cu{nnn_str}',
        'outdir': 'tmp',
        'pseudo_dir': '.',
        'ecutwfc': ecutwfc_ry,
        'ecutrho': ecutrho_ry,
        'occupations': 'smearing',
        'smearing': 'mv',
        'degauss': 0.01,
        'conv_thr': 1e-8,
    }

    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=input_data_scf,
        pseudopotentials=pseudos,
        kpts=kpts_scf,
    )

    # -----------------------------------------------------------------------
    # Bands input (bands.in)
    # -----------------------------------------------------------------------
    input_data_bands = {
        'calculation': 'bands',
        'prefix': f'cu{nnn_str}',
        'outdir': 'tmp',
        'pseudo_dir': '.',
        'ecutwfc': ecutwfc_ry,
        'ecutrho': ecutrho_ry,
        'occupations': 'smearing',
        'smearing': 'mv',
        'degauss': 0.01,
        'conv_thr': 1e-8,
        'nbnd': nbnd,
    }

    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=input_data_bands,
        pseudopotentials=pseudos,
        kpts=kpts_bands,
    )

    # -----------------------------------------------------------------------
    # Bands post-processing input (bands_post.in)
    # -----------------------------------------------------------------------
    bands_post_content = f"""&BANDS
    prefix = 'cu{nnn_str}'
    outdir = 'tmp'
    filband = '{nnn_str}.bands.dat'
    lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)

    print(f"Structure {nnn_str}: wrote pw.in, bands.in, bands_post.in, kpath.json")

print("\nDone. All calculation directories set up.")
