"""
setup_jobs.py - Generate QE input files for all 51 Si structures.

For each structure NNN (000-050):
  - Create calculations/NNN/
  - Copy pseudopotential into calculations/NNN/
  - Write kpath.json from chem_llm.kpath
  - Write pw.in (SCF) and bands.in (NSCF bands) with write_espresso_in
  - Write bands_post.in (bands.x input)

Key decisions (from STEP 4 notes):
  - ecutwfc = 36.0 Ry (18.0 Ha * 2, INVARIANT 5)
  - ecutrho = 144.0 Ry (4 * ecutwfc)
  - occupations = 'fixed' (Si is a semiconductor)
  - conv_thr = 1e-8 Ry
  - nbnd = 8 (4 occupied bands * 2 padding)
  - No spin, no magnetism, no spin-orbit (Si Z=14 < 54)
  - Pseudopotential: Si.pbesol-n-rrjk.upf (scalar relativistic, z_valence=4.0)
"""

import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# Pseudopotential path
PSEUDO_FILE = 'template/Si.pbesol-n-rrjk.upf'

# Read pseudopotential header to get z_valence (INVARIANT 2)
pp_header = read_upf_header(PSEUDO_FILE)
z_valence = pp_header['z_valence']
print(f"Pseudopotential z_valence: {z_valence}")

# Cutoff: 18.0 Ha * 2 = 36.0 Ry (INVARIANT 5)
ECUTWFC_HA = 18.0
ECUTWFC_RY = 2.0 * ECUTWFC_HA  # 36.0 Ry
assert abs(ECUTWFC_RY - 36.0) < 1e-10, f"ecutwfc conversion error: {ECUTWFC_RY}"
ECUTRHO_RY = 4.0 * ECUTWFC_RY  # 144.0 Ry

# nbnd: 2 atoms * 4 valence e- / 2 = 4 occupied bands, pad to 8
N_OCCUPIED = int(2 * z_valence / 2)  # 4
NBND = 2 * N_OCCUPIED  # 8
print(f"N_OCCUPIED={N_OCCUPIED}, NBND={NBND}")

# Common input_data for SCF and bands
COMMON_INPUT = {
    'prefix': 'si',
    'outdir': './tmp',
    'pseudo_dir': '.',
    'ecutwfc': ECUTWFC_RY,
    'ecutrho': ECUTRHO_RY,
    'occupations': 'fixed',
    'conv_thr': 1e-8,
}

PSEUDO_MAP = {'Si': 'Si.pbesol-n-rrjk.upf'}

for i in range(51):
    nnn = f"{i:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(f"{calc_dir}/tmp", exist_ok=True)

    # Copy pseudopotential into calc dir
    shutil.copy(PSEUDO_FILE, f"{calc_dir}/Si.pbesol-n-rrjk.upf")

    # Read structure
    atoms = ase.io.read(f"structures/structure_{nnn}.cif")

    # Get k-path and primitive cell (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)
    kp_save(kp, f"{calc_dir}/kpath.json")

    # Verify nat/ntyp from actual Atoms object (INVARIANT 1)
    nat = len(prim)
    ntyp = len(set(prim.get_chemical_symbols()))
    print(f"Structure {nnn}: nat={nat}, ntyp={ntyp}")
    assert nat == 2, f"Structure {nnn}: expected nat=2, got {nat}"
    assert ntyp == 1, f"Structure {nnn}: expected ntyp=1, got {ntyp}"

    # SCF k-mesh (computed, not hand-typed)
    scf_kpts = scf_kmesh(prim, metal=False)
    print(f"  SCF k-mesh: {scf_kpts}")

    # Write pw.in (SCF)
    scf_input = dict(COMMON_INPUT)
    scf_input['calculation'] = 'scf'
    write_espresso_in(
        f"{calc_dir}/pw.in",
        prim,
        input_data=scf_input,
        pseudopotentials=PSEUDO_MAP,
        kpts=scf_kpts,
    )

    # Write bands.in (NSCF bands)
    bands_input = dict(COMMON_INPUT)
    bands_input['calculation'] = 'bands'
    bands_input['nbnd'] = NBND
    write_espresso_in(
        f"{calc_dir}/bands.in",
        prim,
        input_data=bands_input,
        pseudopotentials=PSEUDO_MAP,
        kpts=kp_kpts(kp),
    )

    # Write bands_post.in (bands.x input)
    # bands.x uses &BANDS namelist: prefix, outdir, filband, lsym
    bands_post_content = f"""&BANDS
  prefix = 'si'
  outdir = './tmp'
  filband = '{nnn}.bands.dat'
  lsym = .false.
/
"""
    with open(f"{calc_dir}/bands_post.in", 'w') as f:
        f.write(bands_post_content)

    print(f"  Wrote {calc_dir}/pw.in, {calc_dir}/bands.in, {calc_dir}/bands_post.in")

print("\nDone: all 51 calculation directories set up.")
