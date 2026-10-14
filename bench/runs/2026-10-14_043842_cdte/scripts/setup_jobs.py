import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# Pseudopotential files (fetched in STEP 3)
PP_Cd = 'template/Cd.pbesol-srn-rrkjus_psl.1.0.0.UPF'
PP_Te = 'template/Te.pbesol-srn-rrkjus_psl.1.0.0.UPF'

# Read z_valence from the actual UPF files (INVARIANT 2)
z_val_Cd = read_upf_header(PP_Cd)['z_valence']
z_val_Te = read_upf_header(PP_Te)['z_valence']
print(f"z_valence: Cd={z_val_Cd}, Te={z_val_Te}")

# Total electrons per formula unit (CdTe)
n_electrons = z_val_Cd + z_val_Te
print(f"Total electrons per formula unit: {n_electrons}")

# Occupied bands (spin-unpolarized, non-metallic)
n_occupied = int(n_electrons / 2)
print(f"Occupied bands: {n_occupied}")

# nbnd: must be >= n_occupied, with room for conduction bands
# Use 2 * n_occupied to be safe
nbnd = 2 * n_occupied
print(f"nbnd: {nbnd}")

# Cutoffs (INVARIANT 5: 1 Ha = 2 Ry)
# Cd recommended: 60 Ha = 120 Ry
# Te recommended: 51 Ha = 102 Ry
# Use max = 120 Ry for ecutwfc
ecutwfc_ry = 120.0
ecutrho_ry = 4 * ecutwfc_ry  # 480 Ry
print(f"ecutwfc: {ecutwfc_ry} Ry, ecutrho: {ecutrho_ry} Ry")

# SCF k-mesh: use scf_kmesh from chem_llm.kpath
# CdTe is a semiconductor, so metal=False

# Pseudopotential mapping for write_espresso_in
pseudopotentials = {
    'Cd': os.path.basename(PP_Cd),
    'Te': os.path.basename(PP_Te),
}

# Common input_data for SCF and bands
common_input = {
    'prefix': 'cdte',
    'outdir': './tmp',
    'pseudo_dir': '.',
    'ecutwfc': ecutwfc_ry,
    'ecutrho': ecutrho_ry,
    'occupations': 'fixed',
    'conv_thr': 1e-8,
    'nspin': 1,
}

# SCF input_data
scf_input = dict(common_input)
scf_input['calculation'] = 'scf'

# Bands input_data
bands_input = dict(common_input)
bands_input['calculation'] = 'bands'
bands_input['nbnd'] = nbnd

# Process each structure
for i in range(51):
    nnn = f"{i:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(f"{calc_dir}/tmp", exist_ok=True)
    
    # Copy pseudopotential files into calculations/NNN/
    shutil.copy(PP_Cd, calc_dir)
    shutil.copy(PP_Te, calc_dir)
    
    # Read the structure
    atoms = ase.io.read(f"structures/structure_{nnn}.cif")
    
    # Get the standard k-path for THIS structure (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    
    # Save kpath.json
    kp_save(kp, f"{calc_dir}/kpath.json")
    
    # SCF k-mesh (computed from the primitive cell)
    kpts_scf = scf_kmesh(prim, metal=False)
    print(f"Structure {nnn}: SCF k-mesh = {kpts_scf}")
    
    # Write pw.in (SCF)
    write_espresso_in(
        f"{calc_dir}/pw.in",
        prim,
        input_data=scf_input,
        pseudopotentials=pseudopotentials,
        kpts=kpts_scf,
    )
    
    # Write bands.in (non-self-consistent bands)
    # kpts = kp_kpts(kp) from the k-path (INVARIANT 3)
    kpts_bands = kp_kpts(kp)
    write_espresso_in(
        f"{calc_dir}/bands.in",
        prim,
        input_data=bands_input,
        pseudopotentials=pseudopotentials,
        kpts=kpts_bands,
    )
    
    # Write bands_post.in (bands.x input)
    # &BANDS: prefix, outdir, filband, lsym=.false.
    bands_post_content = f"""&BANDS
    prefix = 'cdte'
    outdir = './tmp'
    filband = '{nnn}.bands.dat'
    lsym = .false.
/
"""
    with open(f"{calc_dir}/bands_post.in", 'w') as f:
        f.write(bands_post_content)
    
    print(f"Wrote {calc_dir}/pw.in, {calc_dir}/bands.in, {calc_dir}/bands_post.in")

print("Done setting up all 51 calculation directories.")
