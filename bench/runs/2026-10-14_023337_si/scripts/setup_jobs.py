import os
import shutil
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# =============================================================================
# setup_jobs.py
# Creates calculations/NNN/ directories for all 51 structures with:
#   - pseudopotential files
#   - kpath.json
#   - pw.in (SCF)
#   - bands.in (NSCF bands)
#   - bands_post.in (bands.x post-processing)
# =============================================================================

# Pseudopotential file (from template/)
PP_FILE = 'template/Si.pbesol-n-rrkjus_psl.1.0.0.UPF'

# Read z_valence from the actual pseudopotential file (INVARIANT 2)
pp_header = read_upf_header(PP_FILE)
z_valence = pp_header['z_valence']
print(f"Pseudopotential z_valence: {z_valence}")

# Cutoffs (INVARIANT 5): recommended_ecutwfc = 18.0 Ha, 1 Ha = 2 Ry
ecutwfc_ry = 2.0 * 18.0  # 36.0 Ry
ecutrho_ry = 4.0 * ecutwfc_ry  # 144.0 Ry
assert ecutwfc_ry == 36.0, f"ecutwfc_ry = {ecutwfc_ry}, expected 36.0"
assert ecutrho_ry == 144.0, f"ecutrho_ry = {ecutrho_ry}, expected 144.0"

# Common QE settings
common_settings = {
    'prefix': 'si',
    'outdir': './tmp',
    'pseudo_dir': '.',
    'ecutwfc': ecutwfc_ry,
    'ecutrho': ecutrho_ry,
    'occupations': 'fixed',
    'conv_thr': 1e-8,
}

# Pseudopotential mapping for write_espresso_in
pseudopotentials = {'Si': 'Si.pbesol-n-rrkjus_psl.1.0.0.UPF'}

for nnn in range(51):
    calc_dir = f'calculations/{nnn:03d}'
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(f'{calc_dir}/tmp', exist_ok=True)
    
    # Copy pseudopotential into the calculation directory
    shutil.copy(PP_FILE, f'{calc_dir}/Si.pbesol-n-rrkjus_psl.1.0.0.UPF')
    
    # Read the structure
    atoms = ase.io.read(f'structures/structure_{nnn:03d}.cif')
    
    # INVARIANT 1: nat and ntyp computed from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Structure {nnn:03d}: expected 2 atoms, got {nat}"
    assert ntyp == 1, f"Structure {nnn:03d}: expected 1 species, got {ntyp}"
    
    # Get the k-path and primitive cell (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    
    # Save kpath.json
    kp_save(kp, f'{calc_dir}/kpath.json')
    
    # SCF k-point mesh (DECISION RULE: use scf_kmesh, not hand-typed)
    kpts_scf = scf_kmesh(prim, metal=False)
    print(f"Structure {nnn:03d}: SCF kpts = {kpts_scf}")
    
    # Write pw.in (SCF)
    scf_input = dict(common_settings)
    scf_input['calculation'] = 'scf'
    with open(f'{calc_dir}/pw.in', 'w') as f:
        write_espresso_in(f, prim, input_data=scf_input, pseudopotentials=pseudopotentials, kpts=kpts_scf)
    
    # Write bands.in (NSCF bands)
    # nbnd: 2 atoms * 4 valence electrons = 8 electrons -> 4 occupied bands (spin-unpolarized)
    # Pad to 8 to be safe
    nbnd = 8
    bands_input = dict(common_settings)
    bands_input['calculation'] = 'bands'
    bands_input['nbnd'] = nbnd
    kpts_bands = kp_kpts(kp)
    with open(f'{calc_dir}/bands.in', 'w') as f:
        write_espresso_in(f, prim, input_data=bands_input, pseudopotentials=pseudopotentials, kpts=kpts_bands)
    
    # Write bands_post.in (bands.x post-processing)
    # lsym = .false. to skip symmetry analysis (saves time)
    bands_post_content = f"""&BANDS
    prefix = 'si'
    outdir = './tmp'
    filband = '{nnn:03d}.bands.dat'
    lsym = .false.
/
"""
    with open(f'{calc_dir}/bands_post.in', 'w') as f:
        f.write(bands_post_content)
    
    print(f"Structure {nnn:03d}: done")

print("\nAll 51 calculation directories created.")
