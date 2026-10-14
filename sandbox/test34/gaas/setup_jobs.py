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
#   - bands.in (NSCF bands input)
#   - bands_post.in (bands.x post-processing input)
#
# QE settings (from STEP 4 note):
#   ecutwfc = 114 Ry (57 Ha * 2, from Ga pseudopotential, the larger of the two)
#   ecutrho = 456 Ry (4 * ecutwfc)
#   occupations = 'fixed' (GaAs is a semiconductor, band_gap_eV=0.184 from MP)
#   conv_thr = 1e-8 Ry
#   nbnd = 40 (>= 22 occupied bands, with padding)
#   prefix = 'gaas'
#   outdir = './outdir' (relative to calculations/NNN/)
#   pseudo_dir = '.' (pseudopotentials copied into calculations/NNN/)
# =============================================================================

# Pseudopotential files
PP_GA = 'template/Ga.pbesol-rrkjus_psl.1.0.0.UPF'
PP_AS = 'template/As.pbesol-rrkjus_psl.1.0.0.UPF'

# Verify z_valence from the actual UPF files (INVARIANT 2)
pp_ga = read_upf_header(PP_GA)
pp_as = read_upf_header(PP_AS)
z_val_ga = int(pp_ga['z_valence'])
z_val_as = int(pp_as['z_valence'])
print(f"Ga z_valence = {z_val_ga}, As z_valence = {z_val_as}")

# Total electrons per formula unit (GaAs = 1 Ga + 1 As)
n_electrons = z_val_ga + z_val_as
print(f"Total electrons = {n_electrons}")

# Occupied bands (non-spin-polarized, no SOC)
n_occupied = n_electrons // 2
print(f"Occupied bands = {n_occupied}")

# nbnd must be >= n_occupied, with padding
NBND = 40
assert NBND >= n_occupied, f"nbnd={NBND} must be >= n_occupied={n_occupied}"

# Cutoffs (INVARIANT 5: 1 Ha = 2 Ry)
# Ga cutoff = 57 Ha, As cutoff = 52 Ha -> use max = 57 Ha = 114 Ry
ECUTWFC = 114  # Ry
ECUTRHO = 4 * ECUTWFC  # 456 Ry
print(f"ecutwfc = {ECUTWFC} Ry, ecutrho = {ECUTRHO} Ry")

# Pseudopotential filenames for write_espresso_in (relative to calculations/NNN/)
# since pseudo_dir='.' and the UPF files are copied into calculations/NNN/
pseudopotentials = {
    'Ga': 'Ga.pbesol-rrkjus_psl.1.0.0.UPF',
    'As': 'As.pbesol-rrkjus_psl.1.0.0.UPF',
}

# Common input_data (FLAT dict, ASE routes to correct namelists per INVARIANT 6)
common_input = {
    'prefix': 'gaas',
    'outdir': './outdir',
    'pseudo_dir': '.',
    'ecutwfc': ECUTWFC,
    'ecutrho': ECUTRHO,
    'occupations': 'fixed',
    'conv_thr': 1e-8,
}

for i in range(51):
    nnn = f"{i:03d}"
    calc_dir = f'calculations/{nnn}'
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(f'{calc_dir}/outdir', exist_ok=True)
    
    # Copy pseudopotentials into calculations/NNN/
    shutil.copy(PP_GA, calc_dir)
    shutil.copy(PP_AS, calc_dir)
    
    # Read the structure
    atoms = ase.io.read(f'structures/structure_{nnn}.cif')
    
    # Get the standard k-path and primitive cell (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)
    
    # Save kpath.json
    kp_save(kp, f'{calc_dir}/kpath.json')
    
    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(prim)
    ntyp = len(set(prim.get_chemical_symbols()))
    assert nat == 2, f"Structure {nnn}: expected nat=2, got {nat}"
    assert ntyp == 2, f"Structure {nnn}: expected ntyp=2, got {ntyp}"
    
    # SCF k-point mesh (from chem_llm.kpath, not hand-typed)
    kpts_scf = scf_kmesh(prim, metal=False)
    
    # Write pw.in (SCF)
    # write_espresso_in signature: (fd, atoms, input_data, pseudopotentials, kpts, ...)
    scf_input = dict(common_input)
    scf_input['calculation'] = 'scf'
    write_espresso_in(f'{calc_dir}/pw.in', prim, scf_input, pseudopotentials=pseudopotentials, kpts=kpts_scf)
    
    # Write bands.in (NSCF bands)
    bands_input = dict(common_input)
    bands_input['calculation'] = 'bands'
    bands_input['nbnd'] = NBND
    kpts_bands = kp_kpts(kp)
    write_espresso_in(f'{calc_dir}/bands.in', prim, bands_input, pseudopotentials=pseudopotentials, kpts=kpts_bands)
    
    # Write bands_post.in (bands.x input)
    bands_post_content = f"""&BANDS
    prefix = 'gaas'
    outdir = './outdir'
    filband = '{nnn}.bands.dat'
    lsym = .false.
/
"""
    with open(f'{calc_dir}/bands_post.in', 'w') as f:
        f.write(bands_post_content)
    
    if i == 0 or i % 10 == 0:
        print(f"Setup {calc_dir}: nat={nat}, ntyp={ntyp}, kpts_scf={kpts_scf}, n_kpts_bands={len(kpts_bands)}")

print("Done: all 51 calculation directories set up.")
