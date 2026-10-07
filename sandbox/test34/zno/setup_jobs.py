import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save
from chem_llm.tools.upf import read_upf_header

# =============================================================================
# setup_jobs.py
# For each structure NNN (000-050), create calculations/NNN/ containing:
#   - pseudopotential files (Zn.pbesol-sr-pp.upf, O.pbesol-sr-pp.upf)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input)
#   - bands.in (non-self-consistent bands input)
#   - bands_post.in (bands.x post-processing input)
#
# QE settings (see STEP 4 note):
#   ecutwfc = 92 Ry  (46 Ha * 2, from O pseudopotential -- max of Zn=42 Ha, O=46 Ha)
#   ecutrho = 368 Ry (4 * ecutwfc, standard ratio for charge density)
#   occupations = 'fixed' (ZnO is a wide-gap insulator)
#   conv_thr = 1e-8 (dimensionless, in &ELECTRONS)
#   SCF k-mesh: 8x8x4 Monkhorst-Pack (hexagonal cell, a~3.24 A, c~5.22 A)
#   nbnd = 60 (total electrons = 2*20 + 2*6 = 52, occupied bands = 26, pad to 60)
#   prefix = 'zno', outdir = './tmp', pseudo_dir = '.'
# =============================================================================

# --- Read pseudopotential headers to get z_valence (INVARIANT 2) ---
zn_pp = read_upf_header('template/Zn.pbesol-sr-pp.upf')
o_pp = read_upf_header('template/O.pbesol-sr-pp.upf')

z_valence_Zn = zn_pp['z_valence']  # 20.0 (includes semicore 3s2 3p6)
z_valence_O = o_pp['z_valence']    # 6.0

print(f"Zn z_valence = {z_valence_Zn}, O z_valence = {z_valence_O}")

# Total electrons for the 4-atom cell (2 Zn + 2 O)
# This is computed from the actual structure, not hardcoded
base = ase.io.read('structures/structure_000.cif')
n_Zn = sum(1 for s in base.get_chemical_symbols() if s == 'Zn')
n_O = sum(1 for s in base.get_chemical_symbols() if s == 'O')
total_electrons = n_Zn * z_valence_Zn + n_O * z_valence_O
occupied_bands = int(total_electrons / 2)  # spin-unpolarized
print(f"n_Zn = {n_Zn}, n_O = {n_O}, total_electrons = {total_electrons}, occupied_bands = {occupied_bands}")

# nbnd: pad well above occupied_bands to include conduction bands
nbnd = 60
assert nbnd > occupied_bands, f"nbnd={nbnd} must be > occupied_bands={occupied_bands}"

# --- Cutoffs (INVARIANT 5: convert Ha -> Ry) ---
# Zn hint: 42 Ha, O hint: 46 Ha -> use max = 46 Ha = 92 Ry
ecutwfc_ha = max(zn_pp.get('recommended_ecutwfc', {}).get('value', 42.0),
                 o_pp.get('recommended_ecutwfc', {}).get('value', 46.0))
# read_upf_header may not include recommended_ecutwfc; use the known values
# from the get_pseudopotential results: Zn=42 Ha, O=46 Ha
ecutwfc_ha = 46.0  # max of Zn (42 Ha) and O (46 Ha)
ecutwfc_ry = 2.0 * ecutwfc_ha  # 1 Ha = 2 Ry
ecutrho_ry = 4.0 * ecutwfc_ry  # standard 4x ratio
print(f"ecutwfc = {ecutwfc_ry} Ry ({ecutwfc_ha} Ha), ecutrho = {ecutrho_ry} Ry")
assert ecutwfc_ry == 92.0, f"Expected ecutwfc=92 Ry, got {ecutwfc_ry}"
assert ecutrho_ry == 368.0, f"Expected ecutrho=368 Ry, got {ecutrho_ry}"

# --- Pseudopotential filenames (must match files in calculations/NNN/) ---
pseudopotentials = {
    'Zn': 'Zn.pbesol-sr-pp.upf',
    'O': 'O.pbesol-sr-pp.upf',
}

# --- SCF k-mesh (Monkhorst-Pack) ---
# Hexagonal cell: a~3.24 A, c~5.22 A
# 8x8 in-plane gives ~0.19 A^-1 spacing, 4 along c gives ~0.38 A^-1
scf_kpts = (8, 8, 4)

# --- Common input_data for pw.x (flat dict, ASE routes to correct namelists) ---
# INVARIANT 6: do NOT hand-sort into &CONTROL/&SYSTEM/&ELECTRONS
common_input_data = {
    'prefix': 'zno',
    'outdir': './tmp',
    'pseudo_dir': '.',
    'ecutwfc': ecutwfc_ry,
    'ecutrho': ecutrho_ry,
    'occupations': 'fixed',
    'conv_thr': 1e-8,
}

# --- Process each structure ---
for n in range(51):
    nnn = f"{n:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)

    # Copy pseudopotential files into calc_dir (pseudo_dir='.')
    for elem, pp_file in pseudopotentials.items():
        src = f"template/{pp_file}"
        dst = f"{calc_dir}/{pp_file}"
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Read the structure
    atoms = ase.io.read(f"structures/structure_{nnn}.cif")

    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 4, f"Structure {nnn}: expected nat=4, got {nat}"
    assert ntyp == 2, f"Structure {nnn}: expected ntyp=2, got {ntyp}"

    # INVARIANT 3: k-path from chem_llm.kpath for THIS structure
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    kp_save(kp, f"{calc_dir}/kpath.json")

    # --- Write pw.in (SCF) ---
    scf_input_data = dict(common_input_data)
    scf_input_data['calculation'] = 'scf'

    with open(f"{calc_dir}/pw.in", 'w') as f:
        write_espresso_in(
            f, prim, scf_input_data,
            pseudopotentials=pseudopotentials,
            kpts=scf_kpts,
        )

    # --- Write bands.in (non-self-consistent bands) ---
    bands_input_data = dict(common_input_data)
    bands_input_data['calculation'] = 'bands'
    bands_input_data['nbnd'] = nbnd

    # kpts for bands: from kpath (INVARIANT 3)
    bands_kpts = kp_kpts(kp)

    with open(f"{calc_dir}/bands.in", 'w') as f:
        write_espresso_in(
            f, prim, bands_input_data,
            pseudopotentials=pseudopotentials,
            kpts=bands_kpts,
        )

    # --- Write bands_post.in (bands.x input) ---
    # bands.x uses &BANDS namelist with prefix, outdir, filband
    with open(f"{calc_dir}/bands_post.in", 'w') as f:
        f.write("&BANDS\n")
        f.write(f"  prefix = '{common_input_data['prefix']}'\n")
        f.write(f"  outdir = '{common_input_data['outdir']}'\n")
        f.write(f"  filband = '{nnn}.bands.dat'\n")
        f.write("/\n")

    print(f"{nnn}: nat={nat}, ntyp={ntyp}, kpath saved, pw.in, bands.in, bands_post.in written")

print("\nDone: all 51 calculation directories set up.")
