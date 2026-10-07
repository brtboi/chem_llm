import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save
from chem_llm.tools.upf import read_upf_header

# =============================================================================
# STEP 5: setup_jobs.py
# For each structure NNN (000-050), create calculations/NNN/ with:
#   - pseudopotential files (Mo.upf, S.upf)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input, written with write_espresso_in)
#   - bands.in (bands input, written with write_espresso_in)
#   - bands_post.in (bands.x input, written manually)
# =============================================================================

# --- Pseudopotential setup ---
# From STEP 3:
#   Mo: z_valence=14, recommended ecutwfc=40.0 Ha
#   S:  z_valence=6,  recommended ecutwfc=26.0 Ha
# INVARIANT 5: 1 Ha = 2 Ry, so ecutwfc in Ry = 2 * hint_ha
# Mo is the limiting element: ecutwfc = 2 * 40.0 = 80.0 Ry
# ecutrho = 4 * ecutwfc = 320.0 Ry (standard practice)

# Read z_valence from the actual UPF files (INVARIANT 2)
mo_upf = read_upf_header('template/Mo.upf')
s_upf = read_upf_header('template/S.upf')

z_valence_Mo = mo_upf['z_valence']
z_valence_S = s_upf['z_valence']
print(f'Mo z_valence: {z_valence_Mo}')
print(f'S  z_valence: {z_valence_S}')

# Total valence electrons for Mo2S4 (2 Mo + 4 S)
n_Mo = 2
n_S = 4
total_electrons = n_Mo * z_valence_Mo + n_S * z_valence_S
occupied_bands = int(total_electrons / 2)  # spin-unpolarized
print(f'Total valence electrons: {total_electrons}')
print(f'Occupied bands: {occupied_bands}')

# nbnd for bands calculation: pad well above occupied bands
# to capture conduction bands. 40 is well above 26 (14 conduction bands).
# Reduced from 60 to 40 to avoid GPU memory issues with 267 k-points.
nbnd = 40
assert nbnd > occupied_bands, f'nbnd={nbnd} must be > occupied_bands={occupied_bands}'

# ecutwfc in Ry (INVARIANT 5: 1 Ha = 2 Ry)
ecutwfc_ry = 2.0 * 40.0  # Mo is limiting: 40.0 Ha -> 80.0 Ry
ecutrho_ry = 4.0 * ecutwfc_ry  # 320.0 Ry
print(f'ecutwfc = {ecutwfc_ry} Ry')
print(f'ecutrho = {ecutrho_ry} Ry')

# --- SCF k-point mesh ---
# For a 6-atom primitive cell with a=3.19 A, c=13.38 A:
# 12x12x2 gives ~0.25 A^-1 in-plane, ~0.15 A^-1 out-of-plane.
# Reasonable density for a semiconductor band-structure SCF.
scf_kpts = (12, 12, 2)

# --- Common pw.x settings ---
prefix = 'mos2'
outdir = './tmp'
pseudo_dir = '.'  # pseudopotential files are in calculations/NNN/

# --- Pseudopotential filenames (relative to calculations/NNN/) ---
pseudopotentials = {
    'Mo': 'Mo.upf',
    'S': 'S.upf',
}

# --- Number of structures ---
n_structures = 51

for i in range(n_structures):
    nnn = f'{i:03d}'
    calc_dir = f'calculations/{nnn}'
    os.makedirs(calc_dir, exist_ok=True)

    # Copy pseudopotential files into calculations/NNN/
    for elem in ['Mo', 'S']:
        src = f'template/{elem}.upf'
        dst = f'{calc_dir}/{elem}.upf'
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Read the structure
    atoms = ase.io.read(f'structures/structure_{nnn}.cif')

    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 6, f'Structure {nnn}: expected 6 atoms, got {nat}'
    assert ntyp == 2, f'Structure {nnn}: expected 2 species, got {ntyp}'

    # INVARIANT 3: k-path from chem_llm.kpath for THIS structure
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    kpts_bands = kp_kpts(kp)  # k-points for bands.in

    # Save kpath.json (holds labels, ticks, segments)
    kp_save(kp, f'{calc_dir}/kpath.json')

    # --- Write pw.in (SCF) ---
    # INVARIANT 6: pass a FLAT input_data dict; ASE routes each key
    # to its correct namelist via Namelist(...).to_nested(binary='pw')
    scf_input_data = {
        'calculation': 'scf',
        'prefix': prefix,
        'outdir': outdir,
        'pseudo_dir': pseudo_dir,
        'ecutwfc': ecutwfc_ry,
        'ecutrho': ecutrho_ry,
        'occupations': 'fixed',
        'conv_thr': 1.0e-8,
    }

    write_espresso_in(
        f'{calc_dir}/pw.in',
        prim,
        input_data=scf_input_data,
        pseudopotentials=pseudopotentials,
        kpts=scf_kpts,
    )

    # --- Write bands.in (non-self-consistent bands) ---
    bands_input_data = {
        'calculation': 'bands',
        'prefix': prefix,
        'outdir': outdir,
        'pseudo_dir': pseudo_dir,
        'ecutwfc': ecutwfc_ry,
        'ecutrho': ecutrho_ry,
        'occupations': 'fixed',
        'conv_thr': 1.0e-8,
        'nbnd': nbnd,
    }

    write_espresso_in(
        f'{calc_dir}/bands.in',
        prim,
        input_data=bands_input_data,
        pseudopotentials=pseudopotentials,
        kpts=kpts_bands,
    )

    # --- Write bands_post.in (bands.x input) ---
    # bands.x uses a simple namelist &BANDS with prefix, outdir, filband
    filband = f'{nnn}.bands.dat'
    bands_post_content = f"""&BANDS
    prefix = '{prefix}'
    outdir = '{outdir}'
    filband = '{filband}'
/
"""
    with open(f'{calc_dir}/bands_post.in', 'w') as f:
        f.write(bands_post_content)

    if i < 3 or i == n_structures - 1:
        print(f'Set up {calc_dir}/: pw.in, bands.in, bands_post.in, kpath.json, Mo.upf, S.upf')

print(f'\nDone: set up {n_structures} calculation directories (000-050)')
print(f'  ecutwfc = {ecutwfc_ry} Ry, ecutrho = {ecutrho_ry} Ry')
print(f'  SCF kpts = {scf_kpts}')
print(f'  nbnd = {nbnd} (occupied bands = {occupied_bands})')
print(f'  prefix = {prefix}, outdir = {outdir}')
