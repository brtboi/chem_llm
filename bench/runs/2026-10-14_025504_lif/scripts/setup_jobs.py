import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# =============================================================================
# setup_jobs.py
# For each structure NNN (000-050), create calculations/NNN/ containing:
#   - pseudopotential files (copied from template/)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input)
#   - bands.in (non-self-consistent bands input)
#   - bands_post.in (bands.x post-processing input)
#
# All QE inputs are written with write_espresso_in (ASE), not hand-formatted.
# =============================================================================

# --- Pseudopotential info (from get_pseudopotential results) ---
# Li: z_valence=3, sr, recommended_ecutwfc=48 Ha
# F:  z_valence=7, sr, recommended_ecutwfc=42 Ha
# INVARIANT 5: 1 Ha = 2 Ry, so ecutwfc in Ry = 2 * max(48, 42) = 96 Ry
# ecutrho = 4 * ecutwfc = 384 Ry (standard 4x ratio)

li_upf = 'template/Li.upf'
f_upf = 'template/F.upf'

# Read z_valence from the actual UPF files (INVARIANT 2)
li_zv = read_upf_header(li_upf)['z_valence']
f_zv = read_upf_header(f_upf)['z_valence']
print(f'Li z_valence = {li_zv}, F z_valence = {f_zv}')

# Total valence electrons for the 2-atom primitive cell (1 Li + 1 F)
n_electrons = li_zv + f_zv
n_occupied = int(n_electrons / 2)  # spin-unpolarized
print(f'n_electrons = {n_electrons}, n_occupied = {n_occupied}')

# nbnd: pad well above n_occupied to capture conduction bands
# LiF gap ~8.7 eV, so nbnd=40 gives ~30 unoccupied bands, plenty for +/-10 eV window
nbnd = 40
assert nbnd > n_occupied, f'nbnd={nbnd} must exceed n_occupied={n_occupied}'

# --- Cutoffs (INVARIANT 5: convert Ha to Ry) ---
li_ecut_ha = 48.0  # from get_pseudopotential result
f_ecut_ha = 42.0   # from get_pseudopotential result
ecutwfc_ry = 2.0 * max(li_ecut_ha, f_ecut_ha)  # 96 Ry
ecutrho_ry = 4.0 * ecutwfc_ry                   # 384 Ry
print(f'ecutwfc = {ecutwfc_ry} Ry, ecutrho = {ecutrho_ry} Ry')

# --- Pseudopotential filenames for write_espresso_in ---
# pw.x runs inside calculations/NNN, so pseudo_dir must resolve from there.
# We copy the .upf files into calculations/NNN/ and use pseudo_dir='.'
pseudo_files = {'Li': 'Li.upf', 'F': 'F.upf'}

# --- Common input_data (flat dict; ASE routes each key to the correct namelist) ---
# INVARIANT 6: do NOT hand-sort into &CONTROL / &SYSTEM / &ELECTRONS
common_input_data = {
    'prefix': 'lif',
    'outdir': './tmp',
    'pseudo_dir': '.',
    'ecutwfc': ecutwfc_ry,
    'ecutrho': ecutrho_ry,
    'occupations': 'fixed',  # LiF is a clear insulator (gap ~8.7 eV)
    'conv_thr': 1e-8,        # in &ELECTRONS (ASE routes it correctly)
}

# --- Process each structure ---
for i in range(51):
    nnn = f'{i:03d}'
    calc_dir = f'calculations/{nnn}'
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(f'{calc_dir}/tmp', exist_ok=True)

    # Copy pseudopotential files into calc_dir
    for sp, fname in pseudo_files.items():
        src = f'template/{fname}'
        dst = f'{calc_dir}/{fname}'
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Read the structure
    atoms = ase.io.read(f'structures/structure_{nnn}.cif')

    # INVARIANT 1: nat/ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f'{nnn}: expected nat=2, got {nat}'
    assert ntyp == 2, f'{nnn}: expected ntyp=2, got {ntyp}'

    # INVARIANT 3: k-path from chem_llm.kpath for THIS structure
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    kp_save(kp, f'{calc_dir}/kpath.json')

    # SCF k-mesh: computed by scf_kmesh for the primitive cell (insulator, metal=False)
    scf_kpts = scf_kmesh(prim, metal=False)
    print(f'{nnn}: scf_kpts = {scf_kpts}')

    # --- Write pw.in (SCF) ---
    scf_input_data = dict(common_input_data)
    scf_input_data['calculation'] = 'scf'
    write_espresso_in(
        f'{calc_dir}/pw.in',
        prim,
        input_data=scf_input_data,
        pseudopotentials=pseudo_files,
        kpts=scf_kpts,
    )

    # --- Write bands.in (non-self-consistent bands) ---
    bands_input_data = dict(common_input_data)
    bands_input_data['calculation'] = 'bands'
    bands_input_data['nbnd'] = nbnd
    bands_kpts = kp_kpts(kp)  # INVARIANT 3: k-points from the k-path
    write_espresso_in(
        f'{calc_dir}/bands.in',
        prim,
        input_data=bands_input_data,
        pseudopotentials=pseudo_files,
        kpts=bands_kpts,
    )

    # --- Write bands_post.in (bands.x input) ---
    # bands.x uses a different namelist structure; write it directly.
    # &BANDS: prefix, outdir, filband, lsym
    bands_post_content = f"""&BANDS
    prefix = '{common_input_data["prefix"]}'
    outdir = '{common_input_data["outdir"]}'
    filband = '{nnn}.bands.dat'
    lsym = .false.
/
"""
    with open(f'{calc_dir}/bands_post.in', 'w') as f:
        f.write(bands_post_content)

    print(f'{nnn}: done (nat={nat}, ntyp={ntyp}, kpts={len(bands_kpts)})')

print('All 51 calculation directories set up.')
