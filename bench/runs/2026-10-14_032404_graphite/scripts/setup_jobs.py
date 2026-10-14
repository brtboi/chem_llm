"""
setup_jobs.py - Generate QE input files for all 51 graphite structures.

For each structure NNN (000-050), creates calculations/NNN/ containing:
  - C.pbesol-sr-rrkjus_psl.1.0.0.UPF (copied from template/)
  - kpath.json (from chem_llm.kpath)
  - pw.in (SCF input)
  - bands.in (NSCF bands input)
  - bands_post.in (bands.x post-processing input)

Key decisions (see STEP 4 note):
  - ecutwfc = 82.0 Ry (2.0 * 41.0 Ha, INVARIANT 5)
  - ecutrho = 328.0 Ry (4 * ecutwfc)
  - occupations='smearing', smearing='mv', degauss=0.01 Ry (graphite is a semimetal)
  - conv_thr=1.0e-8 Ry
  - nspin=1 (no magnetism)
  - nbnd=16 (16 electrons / 2 = 8 occupied bands, padded to 16)
  - K-mesh: scf_kmesh(prim, metal=True) for SCF
  - Bands k-points: kp_kpts(kp) from chem_llm.kpath
"""

import os
import shutil
import numpy as np
import ase.io
from ase import Atoms
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# Pseudopotential file
PSEUDO_FILE = 'template/C.pbesol-sr-rrkjus_psl.1.0.0.UPF'

# Read the pseudopotential header to get z_valence and verify cutoff
pp_header = read_upf_header(PSEUDO_FILE)
z_valence = pp_header['z_valence']
print(f"Pseudopotential: z_valence = {z_valence}")

# Cutoff conversion (INVARIANT 5): recommended_ecutwfc = 41.0 Ha -> 82.0 Ry
ECUTWFC_HA = 41.0
ECUTWFC_RY = 2.0 * ECUTWFC_HA  # 82.0 Ry
ECUTRHO_RY = 4.0 * ECUTWFC_RY  # 328.0 Ry
assert ECUTWFC_RY == 82.0, f"Expected ecutwfc=82.0 Ry, got {ECUTWFC_RY}"
assert ECUTRHO_RY == 328.0, f"Expected ecutrho=328.0 Ry, got {ECUTRHO_RY}"

# Total electrons and occupied bands
# For the base structure (4 atoms), but we compute per-structure in the loop
# z_valence per atom = 4.0, so 4 atoms * 4 = 16 electrons, 8 occupied bands
# nbnd = 16 (padded above 8)
NBND = 16

# Common input parameters (flat dict, ASE routes to correct namelists - INVARIANT 6)
common_input = {
    'prefix': 'graphite',
    'outdir': './tmp',
    'pseudo_dir': '../../template/',
    'ecutwfc': ECUTWFC_RY,       # Ry
    'ecutrho': ECUTRHO_RY,       # Ry
    'occupations': 'smearing',
    'smearing': 'mv',
    'degauss': 0.01,             # Ry
    'conv_thr': 1.0e-8,          # Ry
    'nspin': 1,
}

N_STRUCTURES = 51

for i in range(N_STRUCTURES):
    nnn = f"{i:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)

    # Copy pseudopotential into the calculation directory
    # (pseudo_dir points to ../../template/, but we also copy for safety)
    # Actually, pseudo_dir='../../template/' means pw.x looks there, so no copy needed.
    # But let's verify the path resolves correctly from calculations/NNN/
    # calculations/NNN/../../template/ = template/  (correct)

    # Read the structure
    atoms = ase.io.read(f"structures/structure_{nnn}.cif")

    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 4, f"Structure {nnn}: expected nat=4, got {nat}"
    assert ntyp == 1, f"Structure {nnn}: expected ntyp=1, got {ntyp}"

    # Get the standard k-path and primitive cell (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)

    # Verify the primitive cell has the same number of atoms
    assert len(prim) == nat, f"Structure {nnn}: prim has {len(prim)} atoms, expected {nat}"

    # Save kpath.json
    kp_save(kp, f"{calc_dir}/kpath.json")

    # SCF k-point mesh (metal=True for semimetal)
    kpts_scf = scf_kmesh(prim, metal=True)
    print(f"Structure {nnn}: SCF k-mesh = {kpts_scf}")

    # Bands k-points from the k-path
    kpts_bands = kp_kpts(kp)
    print(f"Structure {nnn}: Bands k-points = {len(kpts_bands)}")

    # Pseudopotential mapping: species -> filename
    # The filename must match what's in template/
    pseudo_map = {'C': 'C.pbesol-sr-rrkjus_psl.1.0.0.UPF'}

    # Write pw.in (SCF)
    scf_input = dict(common_input)
    scf_input['calculation'] = 'scf'
    write_espresso_in(
        f"{calc_dir}/pw.in",
        prim,
        input_data=scf_input,
        pseudopotentials=pseudo_map,
        kpts=kpts_scf,
    )

    # Write bands.in (NSCF bands)
    bands_input = dict(common_input)
    bands_input['calculation'] = 'bands'
    bands_input['nbnd'] = NBND
    write_espresso_in(
        f"{calc_dir}/bands.in",
        prim,
        input_data=bands_input,
        pseudopotentials=pseudo_map,
        kpts=kpts_bands,
    )

    # Write bands_post.in (bands.x input)
    # bands.x uses a different input format: &BANDS namelist
    bands_post_content = f"""&BANDS
  prefix = 'graphite'
  outdir = './tmp'
  filband = '{nnn}.bands.dat'
  lsym = .false.
/
"""
    with open(f"{calc_dir}/bands_post.in", 'w') as f:
        f.write(bands_post_content)

    print(f"Structure {nnn}: wrote pw.in, bands.in, bands_post.in, kpath.json")

print(f"\nDone: set up {N_STRUCTURES} calculation directories")
