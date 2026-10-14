import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# =============================================================================
# setup_jobs.py
# For each structure NNN, create calculations/NNN/ containing:
#   - Ni pseudopotential file
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input)
#   - bands.in (NSCF bands input)
#   - bands_post.in (bands.x post-processing input)
# =============================================================================

# --- Pseudopotential info ---
# Fetched in STEP 3: Ni.pbesol-rrkjus_psl.1.0.0.UPF
# z_valence = 18.0 (includes semicore 3s2 3p6)
# recommended_ecutwfc = 52.0 Ha = 104.0 Ry (INVARIANT 5: 1 Ha = 2 Ry)
PP_FILE = 'template/Ni.pbesol-rrkjus_psl.1.0.0.UPF'
pp_header = read_upf_header(PP_FILE)
z_valence = pp_header['z_valence']
print(f"Pseudopotential: {PP_FILE}")
print(f"  z_valence = {z_valence}")

# INVARIANT 5: convert cutoff from Ha to Ry
ecutwfc_ha = 52.0  # from get_pseudopotential result
ecutwfc_ry = 2.0 * ecutwfc_ha  # 1 Ha = 2 Ry
ecutrho_ry = 4.0 * ecutwfc_ry  # standard 4x for NC pseudos
print(f"  ecutwfc = {ecutwfc_ha} Ha = {ecutwfc_ry} Ry")
print(f"  ecutrho = {ecutrho_ry} Ry")

# --- Magnetic moment setup ---
# Ni is magnetic (moment 0.707 muB/formula unit from generate_cif)
# nspin=2, set initial moment on Ni atoms
# 0.5 is a fraction of valence electrons (18 for this pseudo)
# -> ~9 muB initial moment, reasonable for Ni
MAGNETIC_SPECIES = ['Ni']
INITIAL_MOMENT = 0.5  # fraction of valence electrons

# --- nbnd for bands calculation ---
# z_valence = 18 for Ni, 1 atom -> 18 electrons total
# With nspin=2, each spin channel has 9 electrons -> 9 occupied bands per channel
# nbnd should be >= 9, use 20 to be safe
NBND = 20
print(f"  nbnd = {NBND} (9 occupied bands per spin channel, padded)")

# --- SCF k-point mesh ---
# Computed via scf_kmesh(prim, metal=True) from chem_llm.kpath
# Ni is a metal -> needs ~2x density

# --- Read base structure ---
base = ase.io.read('structures/structure_000.cif')
print(f"\nBase structure: {len(base)} atoms")

# --- Process each structure ---
for i in range(51):
    nnn = f"{i:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotential into calc dir (pseudo_dir='.')
    pp_dest = os.path.join(calc_dir, os.path.basename(PP_FILE))
    if not os.path.exists(pp_dest):
        shutil.copy(PP_FILE, pp_dest)
    
    # Read structure
    atoms = ase.io.read(f'structures/structure_{nnn}.cif')
    
    # Get k-path and primitive cell (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)
    
    # Set initial magnetic moments on the primitive cell
    # kp_atoms returns a cell with zero moments, so we must set them
    prim.set_initial_magnetic_moments(
        [INITIAL_MOMENT if s in MAGNETIC_SPECIES else 0.0 
         for s in prim.get_chemical_symbols()]
    )
    
    # Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))
    
    # SCF k-point mesh (computed, not hand-typed)
    kpts_scf = scf_kmesh(prim, metal=True)
    print(f"  {nnn}: SCF kpts = {kpts_scf}")
    
    # Pseudopotentials dict: species -> filename (INVARIANT: must be a dict)
    pp_dict = {s: os.path.basename(PP_FILE) for s in prim.get_chemical_symbols()}
    
    # --- Write pw.in (SCF) ---
    # occupations, smearing, degauss belong in &SYSTEM (per QE docs)
    input_data_scf = {
        'control': {
            'calculation': 'scf',
            'prefix': 'ni',
            'outdir': 'tmp',
            'pseudo_dir': '.',
        },
        'system': {
            'ecutwfc': ecutwfc_ry,
            'ecutrho': ecutrho_ry,
            'nspin': 2,
            'occupations': 'smearing',
            'smearing': 'mv',
            'degauss': 0.01,  # Ry
        },
        'electrons': {
            'conv_thr': 1.0e-8,
        },
    }
    
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=input_data_scf,
        pseudopotentials=pp_dict,
        kpts=kpts_scf,
    )
    
    # --- Write bands.in (NSCF bands) ---
    input_data_bands = {
        'control': {
            'calculation': 'bands',
            'prefix': 'ni',
            'outdir': 'tmp',
            'pseudo_dir': '.',
        },
        'system': {
            'ecutwfc': ecutwfc_ry,
            'ecutrho': ecutrho_ry,
            'nspin': 2,
            'nbnd': NBND,
        },
        'electrons': {
            'conv_thr': 1.0e-8,
        },
    }
    
    kpts_bands = kp_kpts(kp)
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=input_data_bands,
        pseudopotentials=pp_dict,
        kpts=kpts_bands,
    )
    
    # --- Write bands_post.in (bands.x input) ---
    bands_post_content = f"""&BANDS
    prefix = 'ni'
    outdir = 'tmp'
    filband = '{nnn}.bands.dat'
    lsym = .false.
    spin_component = 1
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)
    
    print(f"  {nnn}: wrote pw.in, bands.in, bands_post.in, kpath.json")

print("\nDone. All calculation directories created.")
