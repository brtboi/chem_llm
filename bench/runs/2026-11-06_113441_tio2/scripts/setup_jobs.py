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
#   - pseudopotential files (Ti.pbesol_upf, O.pbesol_upf)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input)
#   - bands.in (NSCF bands input)
#   - bands_post.in (bands.x post-processing input)
#
# Key decisions (from STEP 4 note):
#   - ecutwfc = 92.0 Ry (max of Ti 42 Ha, O 46 Ha -> 46 Ha * 2 = 92 Ry)
#   - ecutrho = 368.0 Ry (4 * ecutwfc)
#   - conv_thr = 1.0e-8 Ry
#   - occupations = 'fixed' (rutile TiO2 is a semiconductor, gap ~2.06 eV)
#   - nspin = 1 (no magnetism)
#   - nbnd = 40 (occupied bands = 24 for 6-atom cell; pad well above)
#   - SCF k-mesh: computed via scf_kmesh(prim, metal=False)
#   - Bands k-path: from standard_kpath / kp_kpts (INVARIANT 3)
#   - bands.x: lsym=.false. (skip band-representation file, saves time)
# =============================================================================

# Pseudopotential files
PSEUDO_FILES = {
    'Ti': 'template/Ti.pbesol_upf',
    'O':  'template/O.pbesol_upf',
}

# Read z_valence from the actual pseudopotential files (INVARIANT 2)
# Ti: z_valence=12 (includes semicore 3s2 3p6), O: z_valence=6
# Total electrons per 6-atom cell: 2*12 + 4*6 = 48
# Occupied bands (spin-unpolarized): 48/2 = 24
# nbnd must be >= 24; use 40 for safety
N_BND = 40

# Cutoffs (INVARIANT 5: convert from Ha to Ry, 1 Ha = 2 Ry)
# Ti recommended: 42.0 Ha, O recommended: 46.0 Ha
# Use the maximum: 46.0 Ha * 2 = 92.0 Ry
ECUTWFC_RY = 92.0
ECUTRHO_RY = 4 * ECUTWFC_RY  # 368.0 Ry

# Convergence threshold (Ry)
CONV_THR = 1.0e-8

# Prefix and outdir
PREFIX = 'tio2'
OUTDIR = 'tmp'

# Number of structures
N_STRUCTURES = 51

# Pseudopotential filenames (as they will appear in calculations/NNN/)
PSEUDO_NAMES = {
    'Ti': 'Ti.pbesol_upf',
    'O':  'O.pbesol_upf',
}

for n in range(N_STRUCTURES):
    nnn = f'{n:03d}'
    calc_dir = f'calculations/{nnn}'
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(os.path.join(calc_dir, OUTDIR), exist_ok=True)
    
    # Copy pseudopotential files into calculations/NNN/
    for elem, src in PSEUDO_FILES.items():
        dst = os.path.join(calc_dir, PSEUDO_NAMES[elem])
        shutil.copy2(src, dst)
    
    # Read the structure
    cif_path = f'structures/structure_{nnn}.cif'
    atoms = ase.io.read(cif_path)
    
    # INVARIANT 1: nat/ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 6, f"Structure {nnn}: expected 6 atoms, got {nat}"
    assert ntyp == 2, f"Structure {nnn}: expected 2 species, got {ntyp}"
    
    # Get the k-path and primitive cell (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    
    # Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))
    
    # SCF k-mesh (computed, not hand-typed)
    kpts_scf = scf_kmesh(prim, metal=False)
    
    # Bands k-points from the k-path
    kpts_bands = kp_kpts(kp)
    
    # Pseudopotential mapping for write_espresso_in
    pseudos = {
        'Ti': PSEUDO_NAMES['Ti'],
        'O':  PSEUDO_NAMES['O'],
    }
    
    # --- Write pw.in (SCF) ---
    scf_input_data = {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
        'nspin': 1,
    }
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=scf_input_data,
        pseudopotentials=pseudos,
        kpts=kpts_scf,
    )
    
    # --- Write bands.in (NSCF bands) ---
    bands_input_data = {
        'calculation': 'bands',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'nbnd': N_BND,
        'nspin': 1,
    }
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=bands_input_data,
        pseudopotentials=pseudos,
        kpts=kpts_bands,
    )
    
    # --- Write bands_post.in (bands.x) ---
    # bands.x input: &BANDS namelist with prefix, outdir, filband, lsym
    bands_post_content = f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{nnn}.bands.dat'
    lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)
    
    if n % 10 == 0:
        print(f"  Set up {calc_dir}/ (nat={nat}, ntyp={ntyp}, kpts_scf={kpts_scf}, n_kpts_bands={len(kpts_bands)})")

print(f"\nDone. Set up {N_STRUCTURES} calculation directories.")
