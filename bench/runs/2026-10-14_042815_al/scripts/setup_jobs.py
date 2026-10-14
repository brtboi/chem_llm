import os
import shutil
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# =============================================================================
# setup_jobs.py
# For each structure NNN (000-050), create calculations/NNN/ containing:
#   - pseudopotential files (copied from template/)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input, written with write_espresso_in)
#   - bands.in (NSCF bands input, written with write_espresso_in)
#   - bands_post.in (bands.x post-processing input)
# =============================================================================

# --- Pseudopotential settings (from STEP 3) ---
# Al: nc, pbesol, stringent, UPF, scalar-relativistic
# z_valence = 3.0, recommended_ecutwfc = 20.0 Ha
# INVARIANT 5: ecutwfc_ry = 2.0 * 20.0 = 40.0 Ry
# ecutrho = 4 * ecutwfc = 160.0 Ry
ECUTWFC_RY = 40.0
ECUTRHO_RY = 160.0

# --- SCF settings (from STEP 4 note) ---
# Al is a metal (band_gap_eV=0.0) -> smearing (mv, degauss=0.01 Ry)
# nspin=1 (non-magnetic)
# conv_thr = 1e-8 Ry
# nbnd for bands: z_valence=3.0 per Al atom, 1 atom -> 3 electrons -> 2 occupied bands (spin-unpolarized). Pad nbnd to 8.
NBND = 8

# --- Pseudopotential file mapping ---
# The pseudopotential file is in template/ and will be copied into calculations/NNN/
PSEUDO_FILE = 'Al.pbesol-sr-rrkjus_psl.1.0.0.UPF'
PSEUDO_DIR = './'  # relative to calculations/NNN/

# --- Number of structures ---
N_STRUCTURES = 51

# --- Loop over structures ---
for nnn in range(N_STRUCTURES):
    nnn_str = f'{nnn:03d}'
    calc_dir = f'calculations/{nnn_str}'
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotential into calculations/NNN/
    src_pseudo = f'template/{PSEUDO_FILE}'
    dst_pseudo = f'{calc_dir}/{PSEUDO_FILE}'
    if not os.path.exists(dst_pseudo):
        shutil.copy2(src_pseudo, dst_pseudo)
    
    # Read the structure
    atoms = ase.io.read(f'structures/structure_{nnn_str}.cif')
    
    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 1, f'Structure {nnn_str}: expected nat=1, got {nat}'
    assert ntyp == 1, f'Structure {nnn_str}: expected ntyp=1, got {ntyp}'
    
    # INVARIANT 3: k-path from chem_llm.kpath for THIS structure
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    
    # Save kpath.json
    kp_save(kp, f'{calc_dir}/kpath.json')
    
    # SCF k-mesh: computed via scf_kmesh(prim, metal=True) for a metal
    kpts_scf = scf_kmesh(prim, metal=True)
    
    # --- Write pw.in (SCF) ---
    # Flat input_data dict; ASE routes each key to its correct namelist (INVARIANT 6)
    input_data_scf = {
        'calculation': 'scf',
        'prefix': 'al',
        'outdir': './tmp',
        'pseudo_dir': PSEUDO_DIR,
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'smearing',
        'smearing': 'mv',
        'degauss': 0.01,
        'conv_thr': 1.0e-8,
        'nspin': 1,
    }
    
    pseudos = {'Al': PSEUDO_FILE}
    
    with open(f'{calc_dir}/pw.in', 'w') as f:
        write_espresso_in(f, prim, input_data=input_data_scf, pseudopotentials=pseudos, kpts=kpts_scf)
    
    # --- Write bands.in (NSCF bands) ---
    # calculation='bands', kpts=kp_kpts(kp) per INVARIANT 3
    input_data_bands = {
        'calculation': 'bands',
        'prefix': 'al',
        'outdir': './tmp',
        'pseudo_dir': PSEUDO_DIR,
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'smearing',
        'smearing': 'mv',
        'degauss': 0.01,
        'conv_thr': 1.0e-8,
        'nspin': 1,
        'nbnd': NBND,
    }
    
    with open(f'{calc_dir}/bands.in', 'w') as f:
        write_espresso_in(f, prim, input_data=input_data_bands, pseudopotentials=pseudos, kpts=kp_kpts(kp))
    
    # --- Write bands_post.in (bands.x input) ---
    # &BANDS namelist: prefix, outdir, filband, lsym=.false.
    # lsym=.false. to avoid the slow symmetry analysis (20 min with spin-orbit)
    bands_post_content = f"""&BANDS
    prefix = 'al'
    outdir = './tmp'
    filband = '{nnn_str}.bands.dat'
    lsym = .false.
/
"""
    with open(f'{calc_dir}/bands_post.in', 'w') as f:
        f.write(bands_post_content)
    
    print(f'Structure {nnn_str}: nat={nat}, ntyp={ntyp}, kpts_scf={kpts_scf}, kpath={kp["path"]}')

print('\nDone. All calculation directories created.')
