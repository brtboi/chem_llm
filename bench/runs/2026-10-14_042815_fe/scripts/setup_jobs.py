import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# =============================================================================
# setup_jobs.py
# For each structure NNN (000..050), create calculations/NNN/ with:
#   - pseudopotential files (copied from template/)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input)
#   - bands.in (bands input)
#   - bands_post.in (bands.x post-processing input)
# =============================================================================

# --- Pseudopotential settings -------------------------------------------------
# Fe: nc, sr, pbesol, stringent, UPF
# z_valence = 16 (includes 3s2 3p6 semicore)
# recommended_ecutwfc = 59 Ha = 118 Ry (INVARIANT 5: 1 Ha = 2 Ry)
ECUTWFC_RY = 118.0  # Ry
ECUTRHO_RY = 4 * ECUTWFC_RY  # 4x ecutwfc, standard practice

# --- SCF settings -------------------------------------------------------------
# bcc Fe is a ferromagnetic metal (moment 2.645 muB/f.u., gap 0.0)
# -> nspin=2, smearing (mv, degauss=0.01 Ry), align to Fermi energy
N_SPIN = 2
OCCUPATIONS = 'smearing'
SMearing = 'mv'
DEGAUSS_RY = 0.01  # Ry
CONV_THR = 1e-8

# --- Bands settings -----------------------------------------------------------
# nbnd: z_valence=16, spin-polarized -> ~8 occupied per spin channel
# Set nbnd=40 to safely cover all bands
NBND = 40

# --- Pseudopotential file -----------------------------------------------------
PP_FILE = 'Fe.pbesol-sr-rrkjus_psl.1.0.0.UPF'
PP_DIR = 'template'

# --- Process each structure ---------------------------------------------------
for i in range(51):
    nnn = f'{i:03d}'
    calc_dir = f'calculations/{nnn}'
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotential into calc_dir (pseudo_dir will be '.')
    pp_src = os.path.join(PP_DIR, PP_FILE)
    pp_dst = os.path.join(calc_dir, PP_FILE)
    if not os.path.exists(pp_dst):
        shutil.copy2(pp_src, pp_dst)
    
    # Read structure
    atoms = ase.io.read(f'structures/structure_{nnn}.cif')
    
    # Get k-path and primitive cell (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)
    
    # Set initial magnetic moment on the primitive cell
    # kp_atoms returns a cell with zero moments, so we must set them explicitly
    prim.set_initial_magnetic_moments(
        [0.5 if s == 'Fe' else 0.0 for s in prim.get_chemical_symbols()]
    )
    
    # Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))
    
    # SCF k-mesh (metal=True for ferromagnetic metal)
    kpts_scf = scf_kmesh(prim, metal=True)
    
    # Pseudopotential mapping for write_espresso_in
    pseudopotentials = {'Fe': PP_FILE}
    
    # --- Write pw.in (SCF) ----------------------------------------------------
    input_data_scf = {
        'calculation': 'scf',
        'prefix': 'fe',
        'outdir': './tmp_qe',
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'nspin': N_SPIN,
        'occupations': OCCUPATIONS,
        'smearing': SMearing,
        'degauss': DEGAUSS_RY,
        'conv_thr': CONV_THR,
    }
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=input_data_scf,
        pseudopotentials=pseudopotentials,
        kpts=kpts_scf,
    )
    
    # --- Write bands.in -------------------------------------------------------
    input_data_bands = {
        'calculation': 'bands',
        'prefix': 'fe',
        'outdir': './tmp_qe',
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'nspin': N_SPIN,
        'nbnd': NBND,
    }
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=input_data_bands,
        pseudopotentials=pseudopotentials,
        kpts=kp_kpts(kp),
    )
    
    # --- Write bands_post.in (bands.x) ----------------------------------------
    # &BANDS namelist for bands.x post-processing
    bands_post_content = f"""&BANDS
    prefix = 'fe'
    outdir = './tmp_qe'
    filband = '{nnn}.bands.dat'
    lsym = .false.
    spin_component = 1
/"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)
    
    if i < 3 or i == 50:
        print(f'  {calc_dir}: pw.in, bands.in, bands_post.in, kpath.json written')

print(f'\nDone: set up 51 calculation directories (calculations/000 .. calculations/050)')
print(f'ECUTWFC = {ECUTWFC_RY} Ry, ECUTRHO = {ECUTRHO_RY} Ry')
print(f'N_SPIN = {N_SPIN}, NBND = {NBND}')
