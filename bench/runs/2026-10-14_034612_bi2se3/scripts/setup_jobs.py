#!/usr/bin/env python
"""
setup_jobs.py -- Generate QE input files for all 51 Bi2Se3 structures.

For each structure NNN (000-050), creates calculations/NNN/ containing:
  - Pseudopotential files (Bi, Se) -- pw.x runs inside calculations/NNN/
  - kpath.json -- the standard k-path for this structure
  - pw.in -- SCF input (written with write_espresso_in)
  - bands.in -- NSCF bands input (written with write_espresso_in)
  - bands_post.in -- bands.x post-processing input

Key decisions (see notes):
  - Bi (Z=83) > 54: fully relativistic (fr) pseudopotentials, spin-orbit on
  - noncolin=.true., lspinorb=.true. in &SYSTEM
  - Each band holds ONE electron (spin-orbit), so occupied bands = n_electrons
  - Total electrons: 2*23 (Bi) + 3*24 (Se) = 118
  - nbnd = 256 (generous padding above 118)
  - ecutwfc = 104.0 Ry (from Se's 52.0 Ha * 2), ecutrho = 416.0 Ry
  - occupations='fixed' (narrow-gap semiconductor, not a metal)
  - conv_thr = 1e-8 Ry
  - SCF k-mesh: scf_kmesh(prim, metal=False)
  - Bands k-points: kp_kpts(kp) from the k-path
  - prefix='bi2se3', outdir='./tmp'
  - bands.x: lsym=.false. (skip band-representation file, saves 20 min with SO)
"""

import os
import shutil
import numpy as np
from ase.io import read, write
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# --- Paths ---
STRUCTURES_DIR = 'structures'
CALC_DIR = 'calculations'
TEMPLATE_DIR = 'template'

# Pseudopotential files (relative to calculations/NNN/)
PP_Bi = 'Bi.pbesol-rrkjus_psl.1.0.0.UPF'
PP_Se = 'Se.pbesol-rrkjus_psl.1.0.0.UPF'

# --- Read pseudopotential headers to get z_valence (INVARIANT 2) ---
bi_pp = read_upf_header(os.path.join(TEMPLATE_DIR, PP_Bi))
se_pp = read_upf_header(os.path.join(TEMPLATE_DIR, PP_Se))
z_val_Bi = bi_pp['z_valence']  # 23.0
z_val_Se = se_pp['z_valence']  # 24.0

# Total electrons for Bi2Se3 (2 Bi + 3 Se)
n_electrons = 2 * z_val_Bi + 3 * z_val_Se
print(f"z_valence Bi={z_val_Bi}, Se={z_val_Se}")
print(f"Total electrons: {n_electrons}")

# With spin-orbit, each band holds ONE electron, so occupied bands = n_electrons
n_occupied = int(n_electrons)
print(f"Occupied bands (spin-orbit): {n_occupied}")

# nbnd must be >= n_occupied; use generous padding
nbnd = 256
assert nbnd >= n_occupied, f"nbnd={nbnd} < n_occupied={n_occupied}"

# --- Cutoffs (INVARIANT 5: convert Ha to Ry) ---
# Bi recommended: 44.0 Ha, Se recommended: 52.0 Ha
# Use the larger: 52.0 Ha = 104.0 Ry
ecutwfc_ha = 52.0  # Ha (from Se pseudopotential recommendation)
ecutwfc_ry = 2.0 * ecutwfc_ha  # 104.0 Ry
ecutrho_ry = 4.0 * ecutwfc_ry  # 416.0 Ry
print(f"ecutwfc = {ecutwfc_ry} Ry, ecutrho = {ecutrho_ry} Ry")

# --- Common input parameters ---
PREFIX = 'bi2se3'
OUTDIR = './tmp'

# Pseudopotential filenames (relative to calculations/NNN/)
pseudopotentials = {
    'Bi': PP_Bi,
    'Se': PP_Se,
}

# --- Process each structure ---
for i in range(51):
    nnn = f"{i:03d}"
    calc_dir = os.path.join(CALC_DIR, nnn)
    os.makedirs(calc_dir, exist_ok=True)

    # Copy pseudopotentials into the calculation directory
    for pp_name in [PP_Bi, PP_Se]:
        src = os.path.join(TEMPLATE_DIR, pp_name)
        dst = os.path.join(calc_dir, pp_name)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Read the structure
    atoms = read(os.path.join(STRUCTURES_DIR, f"structure_{nnn}.cif"))

    # Get the standard k-path for THIS structure (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in

    # Save kpath.json (holds labels, ticks, segments)
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))

    # INVARIANT 1: nat/ntyp from the actual ASE Atoms object
    nat = len(prim)
    ntyp = len(set(prim.get_chemical_symbols()))
    print(f"  {nnn}: nat={nat}, ntyp={ntyp}, spacegroup={kp['spacegroup']}")

    # --- SCF k-mesh (DECISION RULE: scf_kmesh, metal=False for semiconductor) ---
    kpts_scf = scf_kmesh(prim, metal=False)
    print(f"  {nnn}: SCF k-mesh = {kpts_scf}")

    # --- Bands k-points from the k-path ---
    kpts_bands = kp_kpts(kp)
    print(f"  {nnn}: Bands k-points = {len(kpts_bands)}")

    # --- SCF input (pw.in) ---
    input_data_scf = {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',  # pseudopotentials are in the same directory
        'ecutwfc': ecutwfc_ry,
        'ecutrho': ecutrho_ry,
        'occupations': 'fixed',
        'conv_thr': 1e-8,
        'noncolin': True,
        'lspinorb': True,
    }

    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=input_data_scf,
        pseudopotentials=pseudopotentials,
        kpts=kpts_scf,
    )

    # --- Bands input (bands.in) ---
    input_data_bands = {
        'calculation': 'bands',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ecutwfc_ry,
        'ecutrho': ecutrho_ry,
        'occupations': 'fixed',
        'conv_thr': 1e-8,
        'noncolin': True,
        'lspinorb': True,
        'nbnd': nbnd,
    }

    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=input_data_bands,
        pseudopotentials=pseudopotentials,
        kpts=kpts_bands,
    )

    # --- bands.x post-processing input (bands_post.in) ---
    # Written manually as a simple namelist file
    bands_post_content = f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{nnn}.bands.dat'
    lsym = .false.
    spin_component = 1
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)

    print(f"  {nnn}: wrote pw.in, bands.in, bands_post.in, kpath.json")

print("\nDone. All calculation directories set up.")
