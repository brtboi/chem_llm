import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# ============================================================
# setup_jobs.py
# For each structure NNN (000-050), create calculations/NNN/ with:
#   - pseudopotential files (Zn.upf, O.upf)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input for pw.x)
#   - bands.in (NSCF bands input for pw.x)
#   - bands_post.in (bands.x post-processing input)
# ============================================================

# --- Pseudopotential cutoffs (INVARIANT 5: convert Ha -> Ry) ---
# Zn: z_valence=20, recommended cutoff 42 Ha (normal hint)
# O:  z_valence=6,  recommended cutoff 46 Ha (normal hint)
# Use the maximum: 46 Ha = 92 Ry (1 Ha = 2 Ry)
Zn_pp = 'template/Zn.upf'
O_pp = 'template/O.upf'

# Read z_valence from the actual UPF files (INVARIANT 2)
z_val_Zn = read_upf_header(Zn_pp)['z_valence']
z_val_O = read_upf_header(O_pp)['z_valence']
print(f"z_valence: Zn={z_val_Zn}, O={z_val_O}")

# Total valence electrons for Zn2O2 (4 atoms: 2 Zn + 2 O)
n_Zn = 2
n_O = 2
total_electrons = n_Zn * z_val_Zn + n_O * z_val_O
print(f"Total valence electrons: {total_electrons}")

# Occupied bands (spin-unpolarized, non-metallic): electrons / 2
n_occupied = int(total_electrons / 2)
print(f"Occupied bands: {n_occupied}")

# nbnd: pad well above occupied bands
nbnd = 60
assert nbnd > n_occupied, f"nbnd={nbnd} must be > n_occupied={n_occupied}"

# ecutwfc in Ry (INVARIANT 5: 1 Ha = 2 Ry)
ecutwfc_ry = 2.0 * 46.0  # 92 Ry
ecutrho_ry = 4.0 * ecutwfc_ry  # 368 Ry
print(f"ecutwfc = {ecutwfc_ry} Ry, ecutrho = {ecutrho_ry} Ry")

# --- Common QE settings ---
prefix = 'zno'
outdir = 'tmp'
pseudo_dir = '.'  # pseudos copied into calculations/NNN/

# --- Pseudopotential mapping ---
pseudopotentials = {'Zn': 'Zn.upf', 'O': 'O.upf'}

# --- Process each structure ---
for n in range(51):
    nnn = f"{n:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)

    # Copy pseudopotentials into the calculation directory
    for pp_file in ['Zn.upf', 'O.upf']:
        src = os.path.join('template', pp_file)
        dst = os.path.join(calc_dir, pp_file)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Read the structure
    atoms = ase.io.read(f"structures/structure_{nnn}.cif")

    # Get the standard k-path for THIS structure (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in

    # Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))

    # --- SCF k-point mesh (DECISION RULE: use scf_kmesh, not hand-typed) ---
    # ZnO is an insulator (gap ~0.73 eV), so metal=False
    scf_kpts = scf_kmesh(prim, metal=False)
    print(f"Structure {nnn}: SCF k-mesh = {scf_kpts}")

    # --- SCF input (pw.in) ---
    scf_input_data = {
        'calculation': 'scf',
        'prefix': prefix,
        'outdir': outdir,
        'pseudo_dir': pseudo_dir,
        'ecutwfc': ecutwfc_ry,
        'ecutrho': ecutrho_ry,
        'occupations': 'fixed',  # insulator
        'conv_thr': 1e-8,
        'nbnd': nbnd,
    }

    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=scf_input_data,
        pseudopotentials=pseudopotentials,
        kpts=scf_kpts,
    )

    # --- Bands input (bands.in) ---
    # NSCF calculation on the k-path (INVARIANT 3: kpts=kp_kpts(kp))
    bands_kpts = kp_kpts(kp)
    bands_input_data = {
        'calculation': 'nscf',
        'prefix': prefix,
        'outdir': outdir,
        'pseudo_dir': pseudo_dir,
        'ecutwfc': ecutwfc_ry,
        'ecutrho': ecutrho_ry,
        'occupations': 'fixed',
        'conv_thr': 1e-8,
        'nbnd': nbnd,
    }

    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=bands_input_data,
        pseudopotentials=pseudopotentials,
        kpts=bands_kpts,
    )

    # --- Bands post-processing input (bands_post.in) ---
    # bands.x input: &BANDS namelist
    bands_post_content = f"""&BANDS
    prefix = '{prefix}'
    outdir = '{outdir}'
    filband = '{nnn}.bands.dat'
    lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)

    print(f"Structure {nnn}: done (nat={len(prim)}, ntyp={len(set(prim.get_chemical_symbols()))})")

print("\nAll 51 calculation directories created.")
