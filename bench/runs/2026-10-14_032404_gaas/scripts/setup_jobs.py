import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# ============================================================
# setup_jobs.py
# Creates calculations/NNN/ directories for each structure with:
#   - pseudopotential files (Ga.upf, As.upf)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input)
#   - bands.in (bands SCF input)
#   - bands_post.in (bands.x post-processing input)
# ============================================================

# Pseudopotential info (from get_pseudopotential results)
# Ga: z_valence=21, recommended_ecutwfc=57.0 Ha
# As: z_valence=23, recommended_ecutwfc=52.0 Ha
# INVARIANT 5: 1 Ha = 2 Ry, so ecutwfc in Ry = 2.0 * max(57.0, 52.0) = 114.0 Ry
# ecutrho = 4 * ecutwfc = 456.0 Ry (standard 4x ratio for charge density)

# Read z_valence from the actual UPF files (INVARIANT 2)
ga_upf = read_upf_header('template/Ga.upf')
as_upf = read_upf_header('template/As.upf')
z_valence_Ga = ga_upf['z_valence']
z_valence_As = as_upf['z_valence']
print(f"Ga z_valence: {z_valence_Ga}")
print(f"As z_valence: {z_valence_As}")

# Total electrons per formula unit (GaAs): 21 + 23 = 44
# Occupied bands (spin-unpolarized): 44 / 2 = 22
n_electrons = int(z_valence_Ga + z_valence_As)
n_occupied = n_electrons // 2
print(f"Total electrons: {n_electrons}, occupied bands: {n_occupied}")

# nbnd for bands: pad above 22, use 48
nbnd = 48
assert nbnd > n_occupied, f"nbnd={nbnd} must be > n_occupied={n_occupied}"

# ecutwfc in Ry (INVARIANT 5: 1 Ha = 2 Ry)
ecutwfc_ha = max(57.0, 52.0)  # max of recommended cutoffs in Ha
ecutwfc_ry = 2.0 * ecutwfc_ha  # convert to Ry
ecutrho_ry = 4.0 * ecutwfc_ry  # standard 4x ratio
print(f"ecutwfc: {ecutwfc_ha} Ha = {ecutwfc_ry} Ry")
print(f"ecutrho: {ecutrho_ry} Ry")

# Pseudopotential filenames for write_espresso_in
# These are relative to pseudo_dir='.' (the calculation directory)
pseudopotentials = {'Ga': 'Ga.upf', 'As': 'As.upf'}

# Common input settings
common_input = {
    'prefix': 'gaas',
    'outdir': 'tmp',
    'pseudo_dir': '.',
    'ecutwfc': ecutwfc_ry,
    'ecutrho': ecutrho_ry,
    'occupations': 'fixed',
    'conv_thr': 1e-8,
}

# Process each structure
for n in range(51):
    nnn = f"{n:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotentials into the calculation directory
    for pp in ['Ga.upf', 'As.upf']:
        src = f"template/{pp}"
        dst = f"{calc_dir}/{pp}"
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    # Read the structure
    atoms = ase.io.read(f"structures/structure_{nnn}.cif")
    
    # Get the standard k-path and primitive cell (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)
    
    # Save kpath.json
    kp_save(kp, f"{calc_dir}/kpath.json")
    
    # INVARIANT 1: compute nat/ntyp from the actual Atoms object
    nat = len(prim)
    ntyp = len(set(prim.get_chemical_symbols()))
    print(f"Structure {nnn}: nat={nat}, ntyp={ntyp}")
    assert nat == 2, f"Expected 2 atoms, got {nat}"
    assert ntyp == 2, f"Expected 2 types, got {ntyp}"
    
    # SCF k-point mesh (from chem_llm.kpath, not hand-typed)
    kpts_scf = scf_kmesh(prim, metal=False)
    print(f"  SCF k-mesh: {kpts_scf}")
    
    # Write pw.in (SCF)
    scf_input = dict(common_input)
    scf_input['calculation'] = 'scf'
    scf_input['nat'] = nat
    scf_input['ntyp'] = ntyp
    
    with open(f"{calc_dir}/pw.in", 'w') as f:
        write_espresso_in(f, prim, input_data=scf_input, pseudopotentials=pseudopotentials, kpts=kpts_scf)
    
    # Write bands.in (non-self-consistent bands)
    bands_input = dict(common_input)
    bands_input['calculation'] = 'bands'
    bands_input['nat'] = nat
    bands_input['ntyp'] = ntyp
    bands_input['nbnd'] = nbnd
    
    # kpts for bands: from the k-path (INVARIANT 3)
    kpts_bands = kp_kpts(kp)
    
    with open(f"{calc_dir}/bands.in", 'w') as f:
        write_espresso_in(f, prim, input_data=bands_input, pseudopotentials=pseudopotentials, kpts=kpts_bands)
    
    # Write bands_post.in (bands.x input)
    # &BANDS: prefix, outdir, filband, lsym=.false.
    with open(f"{calc_dir}/bands_post.in", 'w') as f:
        f.write("&BANDS\n")
        f.write(f"  prefix = '{common_input['prefix']}'\n")
        f.write(f"  outdir = '{common_input['outdir']}'\n")
        f.write(f"  filband = '{nnn}.bands.dat'\n")
        f.write("  lsym = .false.\n")
        f.write("/")
    
    print(f"  Wrote {calc_dir}/pw.in, bands.in, bands_post.in")

print("\nDone. All calculation directories created.")
