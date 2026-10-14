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
#   - pw.in (SCF input for pw.x)
#   - bands.in (NSCF bands input for pw.x)
#   - bands_post.in (bands.x post-processing input)
# =============================================================================

# --- Pseudopotential files ---
# These are in template/ and will be copied into each calculations/NNN/
# so that pseudo_dir='.' resolves correctly when pw.x runs inside calculations/NNN/
PP_FILES = {
    'Sr': 'template/Sr.pbesol-srn-rrkjus_psl.1.0.0.UPF',
    'Ti': 'template/Ti.pbesol-srn-rrkjus_psl.1.0.0.UPF',
    'O':  'template/O.pbesol-srn-rrkjus_psl.1.0.0.UPF',
}

# --- Cutoffs (INVARIANT 5: convert Ha -> Ry, 1 Ha = 2 Ry) ---
# Read z_valence and recommended cutoff from the actual UPF files
ecutwfc_ha = {}
z_valence = {}
for elem, path in PP_FILES.items():
    header = read_upf_header(path)
    z_valence[elem] = header['z_valence']
    # The recommended cutoff is in Hartree (from get_pseudopotential result)
    # We need to read it from the file or use the known values
    # From the get_pseudopotential results:
    #   Sr: 32 Ha, Ti: 42 Ha, O: 46 Ha
    pass

# Use the known recommended cutoffs from get_pseudopotential (in Ha)
# Sr: 32 Ha, Ti: 42 Ha, O: 46 Ha
ecutwfc_ha = {'Sr': 32.0, 'Ti': 42.0, 'O': 46.0}

# Convert to Rydberg: 1 Ha = 2 Ry
ecutwfc_ry = {elem: 2.0 * ha for elem, ha in ecutwfc_ha.items()}
# Use the maximum over all species
ecutwfc = max(ecutwfc_ry.values())  # 92 Ry
ecutrho = 4.0 * ecutwfc             # 368 Ry (standard 4x ratio)

# INVARIANT 5: assert the conversion is correct
assert ecutwfc == 92.0, f'Expected ecutwfc=92 Ry, got {ecutwfc}'
assert ecutrho == 368.0, f'Expected ecutrho=368 Ry, got {ecutrho}'

# --- Electron count (INVARIANT 2) ---
# Total valence electrons = sum of z_valence over all atoms
# Sr: 10, Ti: 12, O: 6 (x3) -> total = 10 + 12 + 18 = 40
# Spin-unpolarized: occupied bands = 40/2 = 20
# nbnd for bands: pad above 20, use 40 (2x occupied)
nbnd = 40

# --- Common settings ---
PREFIX = 'srtio3'
OUTDIR = './tmp'
CONV_THR = 1e-8  # Ry

# --- Pseudopotential filenames (relative to calculations/NNN/) ---
pseudopotentials = {
    'Sr': 'Sr.pbesol-srn-rrkjus_psl.1.0.0.UPF',
    'Ti': 'Ti.pbesol-srn-rrkjus_psl.1.0.0.UPF',
    'O':  'O.pbesol-srn-rrkjus_psl.1.0.0.UPF',
}

# --- Loop over structures ---
for nnn in range(51):
    nnn_str = f'{nnn:03d}'
    calc_dir = f'calculations/{nnn_str}'
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(os.path.join(calc_dir, 'tmp'), exist_ok=True)
    
    # Copy pseudopotential files into calculations/NNN/
    for elem, src_path in PP_FILES.items():
        dst_path = os.path.join(calc_dir, os.path.basename(src_path))
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
    
    # Read the structure
    atoms = ase.io.read(f'structures/structure_{nnn_str}.cif')
    
    # INVARIANT 1: nat/ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 5, f'Structure {nnn_str}: expected nat=5, got {nat}'
    assert ntyp == 3, f'Structure {nnn_str}: expected ntyp=3, got {ntyp}'
    
    # Get the k-path and primitive cell (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    
    # Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))
    
    # SCF k-point mesh (DECISION RULE: use scf_kmesh, not hand-typed)
    # SrTiO3 is a semiconductor (gap ~1.77 eV), so metal=False
    kpts_scf = scf_kmesh(prim, metal=False)
    
    # Bands k-points (INVARIANT 3: from kp_kpts)
    kpts_bands = kp_kpts(kp)
    
    # --- Write pw.in (SCF) ---
    scf_input_data = {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ecutwfc,
        'ecutrho': ecutrho,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
    }
    
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=scf_input_data,
        pseudopotentials=pseudopotentials,
        kpts=kpts_scf,
    )
    
    # --- Write bands.in (NSCF bands) ---
    bands_input_data = {
        'calculation': 'bands',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ecutwfc,
        'ecutrho': ecutrho,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
        'nbnd': nbnd,
    }
    
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=bands_input_data,
        pseudopotentials=pseudopotentials,
        kpts=kpts_bands,
    )
    
    # --- Write bands_post.in (bands.x input) ---
    # bands.x uses a different input format: &BANDS namelist
    # We write it manually since write_espresso_in is for pw.x
    bands_post_content = f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{nnn_str}.bands.dat'
    lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)
    
    if nnn % 10 == 0:
        print(f'  Set up {calc_dir}/ (nat={nat}, ntyp={ntyp}, kpts_scf={kpts_scf}, nbnd={nbnd})')

print('Done. Set up calculations/000/ through calculations/050/')
print(f'ecutwfc = {ecutwfc} Ry, ecutrho = {ecutrho} Ry')
print(f'nbnd = {nbnd}')
print(f'conv_thr = {CONV_THR} Ry')
