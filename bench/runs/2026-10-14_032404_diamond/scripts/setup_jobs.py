import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# =============================================================================
# setup_jobs.py
# For each structure NNN (000-050), create calculations/NNN/ with:
#   - pseudopotential files
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input)
#   - bands.in (NSCF bands input)
#   - bands_post.in (bands.x post-processing input)
# =============================================================================

# --- Pseudopotential ---
# C: Z=6 < 54 -> scalar relativistic (sr), no spin-orbit
# kind=nc, generator=pbesol, accuracy=stringent, format=upf
# Recommended ecutwfc = 41.0 Ha (from get_pseudopotential result)
# INVARIANT 5: 1 Ha = 2 Ry, so ecutwfc = 82.0 Ry
# ecutrho = 4 * ecutwfc = 328.0 Ry (standard 4x factor for charge density)

PSEUDO_DIR = 'template'
PSEUDO_FILE = 'C.pbesol-sr-rrkjus_psl.1.0.0.UPF'

# Read z_valence from the actual UPF file (INVARIANT 2)
upf_path = os.path.join(PSEUDO_DIR, PSEUDO_FILE)
upf_header = read_upf_header(upf_path)
z_valence_C = upf_header['z_valence']
print(f"C pseudopotential: z_valence = {z_valence_C}")

# --- Cutoffs (INVARIANT 5: convert Ha -> Ry) ---
# get_pseudopotential returned recommended_ecutwfc = 41.0 Ha
# 1 Ha = 2 Ry
ECUTWFC_HA = 41.0
ECUTWFC_RY = 2.0 * ECUTWFC_HA  # 82.0 Ry
ECUTRHO_RY = 4.0 * ECUTWFC_RY  # 328.0 Ry

assert abs(ECUTWFC_RY - 82.0) < 1e-10, f"ecutwfc conversion error: {ECUTWFC_RY}"
assert abs(ECUTRHO_RY - 328.0) < 1e-10, f"ecutrho conversion error: {ECUTRHO_RY}"

# --- Electron count and nbnd (INVARIANT 2) ---
# Diamond C: 2 atoms x 4 valence electrons = 8 electrons total
# Spin-unpolarized (nspin=1), insulator -> occupied bands = 8/2 = 4
# nbnd must be > occupied bands; use 16 for room for conduction bands
N_ATOMS = 2  # will be verified per structure
N_OCCUPIED = int(N_ATOMS * z_valence_C / 2)
NBND = 16
assert NBND > N_OCCUPIED, f"nbnd={NBND} must be > n_occupied={N_OCCUPIED}"
print(f"Electron count: {N_ATOMS} x {z_valence_C} = {N_ATOMS * z_valence_C} electrons")
print(f"Occupied bands: {N_OCCUPIED}, nbnd: {NBND}")

# --- QE settings ---
PREFIX = 'pwscf'
OUTDIR = './tmp'
CONV_THR = 1e-8  # Ry (in &ELECTRONS, not &CONTROL -- INVARIANT 6)

# --- Loop over structures ---
for i in range(51):
    nnn = f"{i:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(os.path.join(calc_dir, OUTDIR), exist_ok=True)

    # Copy pseudopotential into calc_dir (pseudo_dir='.' resolves from here)
    src_pseudo = os.path.join(PSEUDO_DIR, PSEUDO_FILE)
    dst_pseudo = os.path.join(calc_dir, PSEUDO_FILE)
    if not os.path.exists(dst_pseudo):
        shutil.copy2(src_pseudo, dst_pseudo)

    # Read structure
    cif_path = f"structures/structure_{nnn}.cif"
    atoms = ase.io.read(cif_path)

    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == N_ATOMS, f"Structure {nnn}: nat={nat}, expected {N_ATOMS}"
    assert ntyp == 1, f"Structure {nnn}: ntyp={ntyp}, expected 1"

    # --- K-path (INVARIANT 3) ---
    # For structure_000 (unperturbed), use the standard k-path from seekpath.
    # For perturbed structures, the symmetry is broken, so standard_kpath will
    # return a lower-symmetry path. We use it per structure.
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))

    # INVARIANT 1: nat/ntyp from the primitive cell (the one written into QE inputs)
    nat_prim = len(prim)
    ntyp_prim = len(set(prim.get_chemical_symbols()))
    assert nat_prim == nat, f"Structure {nnn}: prim nat={nat_prim} != {nat}"
    assert ntyp_prim == ntyp, f"Structure {nnn}: prim ntyp={ntyp_prim} != {ntyp}"

    # --- SCF k-point mesh (DECISION RULE: never typed by hand) ---
    # Diamond is an insulator -> metal=False
    kpts_scf = scf_kmesh(prim, metal=False)
    print(f"Structure {nnn}: SCF k-mesh = {kpts_scf}")

    # --- Pseudopotential mapping for write_espresso_in ---
    # write_espresso_in expects {element: filename} where filename is relative
    # to pseudo_dir. Since we copied the pseudo into calc_dir and pseudo_dir='.',
    # the filename is just the basename.
    pseudo_map = {'C': PSEUDO_FILE}

    # --- SCF input (pw.in) ---
    input_data_scf = {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',  # diamond is an insulator
        'conv_thr': CONV_THR,    # in &ELECTRONS (INVARIANT 6)
        'nspin': 1,              # non-magnetic
    }

    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=input_data_scf,
        pseudopotentials=pseudo_map,
        kpts=kpts_scf,
    )

    # --- Bands input (bands.in) ---
    # calculation='bands' -> non-self-consistent band structure
    # kpts = kp_kpts(kp) from the k-path (INVARIANT 3)
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
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=input_data_bands,
        pseudopotentials=pseudo_map,
        kpts=kp_kpts(kp),
    )

    # --- Bands post-processing input (bands_post.in) ---
    # bands.x input: &BANDS namelist
    # lsym=.false. to skip symmetry analysis (saves time)
    bands_post_content = f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{nnn}.bands.dat'
    lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)

    if i == 0 or i % 10 == 0:
        print(f"Wrote calculations/{nnn}/ (pw.in, bands.in, bands_post.in, kpath.json)")

print("Done: wrote calculations/000 through calculations/050")
