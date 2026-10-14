import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# =============================================================================
# setup_jobs.py
# Creates calculations/NNN/ for each structure NNN (000-050) with:
#   - Pseudopotential files (copied from template/)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input)
#   - bands.in (NSCF bands input)
#   - bands_post.in (bands.x post-processing input)
#
# QE settings (see STEP 4 note):
#   - ecutwfc = 92 Ry (46 Ha * 2, from O pseudopotential, the larger of Zn/O)
#   - ecutrho = 4 * ecutwfc = 368 Ry
#   - occupations = 'fixed' (ZnO is a wide-gap insulator)
#   - conv_thr = 1e-8 Ry
#   - nspin = 1 (non-magnetic)
#   - nbnd = 64 (well above 26 occupied bands)
#   - prefix = 'zno'
#   - outdir = './tmp'
#   - pseudo_dir = '.' (pseudopotentials are in the same directory as pw.in)
# =============================================================================

# Pseudopotential files
pp_files = {
    'Zn': 'template/Zn.pbesol-n-rrkjus_psl.1.0.0.UPF',
    'O': 'template/O.pbesol-n-rrkjus_psl.1.0.0.UPF',
}

# Read z_valence from pseudopotential files (INVARIANT 2)
z_valences = {}
for elem, pp_path in pp_files.items():
    header = read_upf_header(pp_path)
    z_valences[elem] = header['z_valence']
    print(f"{elem}: z_valence = {z_valences[elem]}")

# Total electrons for the primitive cell (2 Zn + 2 O)
n_electrons = 2 * z_valences['Zn'] + 2 * z_valences['O']
print(f"Total electrons: {n_electrons}")
n_occupied = int(n_electrons / 2)  # spin-unpolarized
print(f"Occupied bands: {n_occupied}")

# nbnd must be > n_occupied
nbnd = 64
assert nbnd > n_occupied, f"nbnd={nbnd} must be > n_occupied={n_occupied}"

# Cutoffs (INVARIANT 5: convert from Ha to Ry)
# Zn: 42 Ha, O: 46 Ha -> use max = 46 Ha = 92 Ry
ecutwfc_ha = max(42.0, 46.0)  # from get_pseudopotential results
ecutwfc_ry = 2.0 * ecutwfc_ha  # 1 Ha = 2 Ry
ecutrho_ry = 4.0 * ecutwfc_ry
print(f"ecutwfc = {ecutwfc_ry} Ry, ecutrho = {ecutrho_ry} Ry")

# SCF input data (flat dict, ASE routes to correct namelists per INVARIANT 6)
scf_input_data = {
    'calculation': 'scf',
    'prefix': 'zno',
    'outdir': './tmp',
    'pseudo_dir': '.',  # pseudopotentials are in the same directory as pw.in
    'ecutwfc': ecutwfc_ry,
    'ecutrho': ecutrho_ry,
    'occupations': 'fixed',
    'conv_thr': 1e-8,
    'nspin': 1,
}

# Bands input data
bands_input_data = {
    'calculation': 'bands',
    'prefix': 'zno',
    'outdir': './tmp',
    'pseudo_dir': '.',  # pseudopotentials are in the same directory as bands.in
    'ecutwfc': ecutwfc_ry,
    'ecutrho': ecutrho_ry,
    'occupations': 'fixed',
    'conv_thr': 1e-8,
    'nspin': 1,
    'nbnd': nbnd,
}

# Pseudopotential filenames (relative to calculations/NNN/)
pp_dict = {
    'Zn': 'Zn.pbesol-n-rrkjus_psl.1.0.0.UPF',
    'O': 'O.pbesol-n-rrkjus_psl.1.0.0.UPF',
}

for nnn in range(51):
    calc_dir = f'calculations/{nnn:03d}'
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotential files
    for elem, pp_path in pp_files.items():
        shutil.copy2(pp_path, os.path.join(calc_dir, pp_dict[elem]))
    
    # Read the structure
    atoms = ase.io.read(f'structures/structure_{nnn:03d}.cif')
    
    # Get the k-path and primitive cell (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)
    
    # Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))
    
    # Compute SCF k-mesh (DECISION RULE: use scf_kmesh, not hand-typed)
    kpts_scf = scf_kmesh(prim, metal=False)
    
    # Write pw.in (SCF)
    with open(os.path.join(calc_dir, 'pw.in'), 'w') as f:
        write_espresso_in(f, prim, input_data=scf_input_data,
                          pseudopotentials=pp_dict, kpts=kpts_scf)
    
    # Write bands.in (NSCF bands)
    kpts_bands = kp_kpts(kp)
    with open(os.path.join(calc_dir, 'bands.in'), 'w') as f:
        write_espresso_in(f, prim, input_data=bands_input_data,
                          pseudopotentials=pp_dict, kpts=kpts_bands)
    
    # Write bands_post.in (bands.x input)
    # lsym=.false. to skip symmetry analysis (saves time, not needed here)
    bands_post = f"""&BANDS
  prefix = 'zno'
  outdir = './tmp'
  filband = '{nnn:03d}.bands.dat'
  lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post)
    
    # INVARIANT 1: verify nat/ntyp from the actual Atoms object
    nat = len(prim)
    ntyp = len(set(prim.get_chemical_symbols()))
    assert nat == 4, f"Structure {nnn:03d}: expected 4 atoms, got {nat}"
    assert ntyp == 2, f"Structure {nnn:03d}: expected 2 species, got {ntyp}"
    
    if nnn == 0:
        print(f"  {calc_dir}: SCF k-mesh = {kpts_scf}, bands k-points = {len(kpts_bands)}")

print(f"\nSetup complete for 51 structures in calculations/")
print("Done.")
