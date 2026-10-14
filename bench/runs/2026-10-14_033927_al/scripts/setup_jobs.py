import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# ============================================================
# setup_jobs.py
# For each structure NNN (000-050), create calculations/NNN/
# with: pseudopotential files, kpath.json, pw.in (SCF),
#       bands.in (NSCF bands), bands_post.in (bands.x post).
# ============================================================

# --- Pseudopotential ---
PP_FILE = 'template/Al.pbesol-sr-rrkjus_psl.1.0.0.UPF'
pp_header = read_upf_header(PP_FILE)
z_valence = pp_header['z_valence']  # 3.0 for Al
print(f'Pseudopotential: {PP_FILE}, z_valence={z_valence}')

# Pseudopotential filename for the pseudopotentials dict passed to write_espresso_in
PP_NAME = os.path.basename(PP_FILE)  # 'Al.pbesol-sr-rrkjus_psl.1.0.0.UPF'

# --- Cutoffs (INVARIANT 5: convert Ha -> Ry) ---
# get_pseudopotential recommended ecutwfc = 20.0 Ha
# 1 Ha = 2 Ry, so ecutwfc = 40.0 Ry
ecutwfc_ha = 20.0  # from get_pseudopotential result
ecutwfc_ry = 2.0 * ecutwfc_ha  # 40.0 Ry
ecutrho_ry = 4.0 * ecutwfc_ry  # 160.0 Ry (standard 4x ratio)
assert ecutwfc_ry == 40.0, f'ecutwfc_ry = {ecutwfc_ry}, expected 40.0'
assert ecutrho_ry == 160.0, f'ecutrho_ry = {ecutrho_ry}, expected 160.0'

# --- Common settings ---
PREFIX = 'al'
OUTDIR = './tmp'
PSEUDO_DIR = '.'  # pseudopotential files copied into calculations/NNN

# Al is a metal (band_gap_eV = 0.0), so:
# - occupations = 'smearing', smearing = 'mv', degauss = 0.01 Ry
# - nspin = 1 (non-magnetic, moment = 0.0)
# - conv_thr = 1e-8

# --- nbnd for bands ---
# Al: 1 atom, z_valence = 3.0, so 3 electrons total.
# Spin-unpolarized: occupied bands = 3/2 = 1.5, so nbnd >= 2.
# Set nbnd = 10 (generous padding).
NBND = 10

# --- Loop over structures ---
for i in range(51):
    nnn = f'{i:03d}'
    calc_dir = f'calculations/{nnn}'
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(os.path.join(calc_dir, 'tmp'), exist_ok=True)
    
    # Copy pseudopotential into calc_dir (pseudo_dir = '.')
    pp_dest = os.path.join(calc_dir, PP_NAME)
    if not os.path.exists(pp_dest):
        shutil.copy2(PP_FILE, pp_dest)
    
    # Read structure
    atoms = ase.io.read(f'structures/structure_{nnn}.cif')
    
    # INVARIANT 1: nat and ntyp from actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 1, f'structure_{nnn}: expected 1 atom, got {nat}'
    assert ntyp == 1, f'structure_{nnn}: expected 1 species, got {ntyp}'
    
    # INVARIANT 3: k-path from chem_llm.kpath for THIS structure
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    
    # Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))
    
    # SCF k-mesh (metal=True for Al)
    kpts_scf = scf_kmesh(prim, metal=True)
    
    # Bands k-points from kpath
    kpts_bands = kp_kpts(kp)
    
    # Pseudopotentials dict for write_espresso_in
    pseudopotentials = {'Al': PP_NAME}
    
    # --- Write pw.in (SCF) ---
    input_data_scf = {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': PSEUDO_DIR,
        'ecutwfc': ecutwfc_ry,
        'ecutrho': ecutrho_ry,
        'occupations': 'smearing',
        'smearing': 'mv',
        'degauss': 0.01,
        'conv_thr': 1.0e-8,
        'nspin': 1,
    }
    write_espresso_in(os.path.join(calc_dir, 'pw.in'), prim, input_data_scf, pseudopotentials=pseudopotentials, kpts=kpts_scf)
    
    # --- Write bands.in (NSCF bands) ---
    input_data_bands = {
        'calculation': 'bands',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': PSEUDO_DIR,
        'ecutwfc': ecutwfc_ry,
        'ecutrho': ecutrho_ry,
        'occupations': 'smearing',
        'smearing': 'mv',
        'degauss': 0.01,
        'conv_thr': 1.0e-8,
        'nspin': 1,
        'nbnd': NBND,
    }
    write_espresso_in(os.path.join(calc_dir, 'bands.in'), prim, input_data_bands, pseudopotentials=pseudopotentials, kpts=kpts_bands)
    
    # --- Write bands_post.in (bands.x post-processing) ---
    # bands.x input format: &BANDS namelist
    bands_post_content = f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{nnn}.bands.dat'
    lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)
    
    print(f'{nnn}: nat={nat}, ntyp={ntyp}, kpts_scf={kpts_scf}, n_kpts_bands={len(kpts_bands)}')

print('Done: all calculation directories set up.')
