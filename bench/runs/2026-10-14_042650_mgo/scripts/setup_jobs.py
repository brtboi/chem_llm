import os
import shutil
import ase.io
import ase.io.espresso
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# ============================================================
# setup_jobs.py
# For each structure NNN (000..050), create calculations/NNN/ with:
#   - pseudopotential files (Mg.upf, O.upf)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input, written with write_espresso_in)
#   - bands.in (NSCF bands input, written with write_espresso_in)
#   - bands_post.in (bands.x post-processing input)
# ============================================================

# --- Pseudopotential cutoffs (INVARIANT 5) ---
# Mg.upf: recommended_ecutwfc = 42.0 Ha
# O.upf:  recommended_ecutwfc = 46.0 Ha
# Use the higher (O) cutoff for the whole calculation.
# 1 Ha = 2 Ry, so:
ecutwfc_ry = 2.0 * 46.0  # = 92 Ry
ecutrho_ry = 4.0 * ecutwfc_ry  # = 368 Ry (standard 4x ratio)
assert ecutwfc_ry == 92.0, f"ecutwfc_ry = {ecutwfc_ry}, expected 92.0"
assert ecutrho_ry == 368.0, f"ecutrho_ry = {ecutrho_ry}, expected 368.0"

# --- Valence electron count (INVARIANT 2) ---
# Mg: z_valence = 10 (includes semicore 3s2 3p6)
# O:  z_valence = 6
# Total = 16 electrons -> 8 occupied bands (spin-unpolarized)
# nbnd must be >= 8; use 16 for safety.
nbnd = 16

# --- SCF k-point mesh ---
# For the primitive rock-salt cell (a ~ 2.97 A), a 6x6x6 Monkhorst-Pack grid
# gives ~0.35 A^-1 spacing, adequate for an insulator with a ~4 eV gap.
# scf_kmesh(prim, metal=False) computes this automatically.

# --- Common input_data for pw.x (flat dict, ASE routes to correct namelists) ---
# INVARIANT 6: do NOT hand-sort into &CONTROL/&SYSTEM/&ELECTRONS.
# ASE's write_espresso_in handles placement.
common_input_data = {
    'prefix': 'mgo',
    'outdir': './tmp',
    'pseudo_dir': '.',
    'ecutwfc': ecutwfc_ry,   # Ry
    'ecutrho': ecutrho_ry,   # Ry
    'occupations': 'fixed',  # insulator (band gap ~4.4 eV)
    'conv_thr': 1.0e-8,      # Ry, in &ELECTRONS (ASE routes it)
    'nspin': 1,              # non-magnetic
}

# Pseudopotential filenames (relative to calculations/NNN/)
pseudopotentials = {
    'Mg': 'Mg.upf',
    'O': 'O.upf',
}

# --- Loop over all 51 structures ---
for i in range(51):
    nnn = f"{i:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)

    # Copy pseudopotentials into the calculation directory
    for pp_file in ['Mg.upf', 'O.upf']:
        src = f"template/{pp_file}"
        dst = f"{calc_dir}/{pp_file}"
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Read the structure
    atoms = ase.io.read(f"structures/structure_{nnn}.cif")

    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Structure {nnn}: nat={nat}, expected 2"
    assert ntyp == 2, f"Structure {nnn}: ntyp={ntyp}, expected 2"

    # INVARIANT 3: k-path from chem_llm.kpath for THIS structure
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    kp_save(kp, f"{calc_dir}/kpath.json")

    # SCF k-point mesh (computed, not hand-typed)
    kpts_scf = scf_kmesh(prim, metal=False)

    # --- Write pw.in (SCF) ---
    scf_input_data = dict(common_input_data)
    scf_input_data['calculation'] = 'scf'
    ase.io.espresso.write_espresso_in(
        f"{calc_dir}/pw.in",
        prim,
        input_data=scf_input_data,
        pseudopotentials=pseudopotentials,
        kpts=kpts_scf,
    )

    # --- Write bands.in (NSCF bands) ---
    bands_input_data = dict(common_input_data)
    bands_input_data['calculation'] = 'bands'
    bands_input_data['nbnd'] = nbnd
    ase.io.espresso.write_espresso_in(
        f"{calc_dir}/bands.in",
        prim,
        input_data=bands_input_data,
        pseudopotentials=pseudopotentials,
        kpts=kp_kpts(kp),  # INVARIANT 3: k-points from the k-path
    )

    # --- Write bands_post.in (bands.x input) ---
    # &BANDS namelist: prefix, outdir, filband, lsym
    # lsym=.false. to skip band-representation file (not needed, saves time)
    bands_post_content = f"""&BANDS
    prefix = 'mgo'
    outdir = './tmp'
    filband = '{nnn}.bands.dat'
    lsym = .false.
/
"""
    with open(f"{calc_dir}/bands_post.in", 'w') as f:
        f.write(bands_post_content)

    print(f"{nnn}: done (nat={nat}, ntyp={ntyp}, kpts_scf={kpts_scf}, nbnd={nbnd})")

print("\nAll 51 calculation directories created.")
print("Next: validate calculations/000/pw.in and calculations/000/bands.in")
