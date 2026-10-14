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
#   - C.pbesol-sr-rrkjus_psl.1.0.0.UPF (pseudopotential, copied from template/)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input)
#   - bands.in (NSCF bands input)
#   - bands_post.in (bands.x post-processing input)
#
# Key decisions (from STEP 4 note):
#   - ecutwfc = 82.0 Ry (41.0 Ha * 2, INVARIANT 5)
#   - ecutrho = 328.0 Ry (4 * ecutwfc)
#   - occupations = 'fixed' (diamond C is an insulator, gap ~4.1 eV)
#   - conv_thr = 1e-8 Ry (in &ELECTRONS, routed by ASE)
#   - SCF k-mesh from scf_kmesh(prim, metal=False)
#   - Bands: nbnd=8 (4 occupied bands, padded), kpts=kp_kpts(kp)
#   - Cell for both pw.in and bands.in: kp_atoms(kp) (INVARIANT 3)
#   - bands.x: &BANDS prefix='scf', outdir='tmp_scf', filband='NNN.bands.dat', lsym=.false.
# =============================================================================

# Pseudopotential file
PP_FILE = 'C.pbesol-sr-rrkjus_psl.1.0.0.UPF'
PP_PATH = os.path.join('template', PP_FILE)

# Read the pseudopotential header to get z_valence (INVARIANT 2)
pp_header = read_upf_header(PP_PATH)
z_valence_C = pp_header['z_valence']
print(f"C pseudopotential z_valence = {z_valence_C}")

# Cutoffs (INVARIANT 5: 1 Ha = 2 Ry)
ecutwfc_ha = 41.0  # from get_pseudopotential recommended_ecutwfc
ecutwfc_ry = 2.0 * ecutwfc_ha  # 82.0 Ry
ecutrho_ry = 4.0 * ecutwfc_ry  # 328.0 Ry
assert ecutwfc_ry == 82.0, f"Expected ecutwfc=82.0 Ry, got {ecutwfc_ry}"
assert ecutrho_ry == 328.0, f"Expected ecutrho=328.0 Ry, got {ecutrho_ry}"

# Number of occupied bands: 2 atoms * 4 valence e / 2 = 4
# Pad to 8 for the bands calculation
n_occupied = int(len(set([1])) * 0)  # placeholder, will compute per structure

# Common input data for SCF and bands
common_input = {
    'ecutwfc': ecutwfc_ry,
    'ecutrho': ecutrho_ry,
    'occupations': 'fixed',
    'conv_thr': 1e-8,
}

# Pseudopotential mapping for write_espresso_in
pseudo_map = {'C': PP_FILE}

for nnn in range(51):
    calc_dir = f'calculations/{nnn:03d}'
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotential into the calculation directory
    shutil.copy2(PP_PATH, os.path.join(calc_dir, PP_FILE))
    
    # Read the structure
    atoms = ase.io.read(f'structures/structure_{nnn:03d}.cif')
    
    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Structure {nnn:03d}: expected nat=2, got {nat}"
    assert ntyp == 1, f"Structure {nnn:03d}: expected ntyp=1, got {ntyp}"
    
    # Get the standard k-path for THIS structure (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    
    # Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))
    
    # SCF k-mesh (computed, not typed by hand)
    kpts_scf = scf_kmesh(prim, metal=False)
    print(f"Structure {nnn:03d}: SCF k-mesh = {kpts_scf}")
    
    # Write pw.in (SCF)
    scf_input = dict(common_input)
    scf_input['calculation'] = 'scf'
    scf_input['prefix'] = 'scf'
    scf_input['outdir'] = 'tmp_scf'
    scf_input['pseudo_dir'] = '.'
    
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=scf_input,
        pseudopotentials=pseudo_map,
        kpts=kpts_scf,
    )
    
    # Write bands.in (NSCF bands)
    # nbnd: 4 occupied bands (2 atoms * 4 valence e / 2), pad to 8
    nbnd = 8
    bands_input = dict(common_input)
    bands_input['calculation'] = 'bands'
    bands_input['prefix'] = 'scf'  # same as SCF, reads SCF wavefunctions
    bands_input['outdir'] = 'tmp_scf'
    bands_input['pseudo_dir'] = '.'
    bands_input['nbnd'] = nbnd
    
    # kpts for bands: the k-path points from chem_llm.kpath (INVARIANT 3)
    kpts_bands = kp_kpts(kp)
    
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=bands_input,
        pseudopotentials=pseudo_map,
        kpts=kpts_bands,
    )
    
    # Write bands_post.in (bands.x input)
    # &BANDS namelist: prefix, outdir, filband, lsym
    bands_post_content = f"""&BANDS
    prefix = 'scf'
    outdir = 'tmp_scf'
    filband = '{nnn:03d}.bands.dat'
    lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)
    
    if nnn <= 2 or nnn == 50:
        print(f"  Wrote {calc_dir}/: pw.in, bands.in, bands_post.in, kpath.json, {PP_FILE}")

print("\nDone. Generated calculation directories for all 51 structures.")
print(f"ecutwfc = {ecutwfc_ry} Ry, ecutrho = {ecutrho_ry} Ry")
print(f"nbnd = 8 (4 occupied bands, padded)")
