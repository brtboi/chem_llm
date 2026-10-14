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
#   - pseudopotential files (Mo.upf, S.upf)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input)
#   - bands.in (NSCF bands input)
#   - bands_post.in (bands.x post-processing input)
# =============================================================================

# Pseudopotential paths
PSEUDO_DIR = 'template'

# Read pseudopotential headers to get z_valence and recommended cutoffs
mo_upf = read_upf_header(os.path.join(PSEUDO_DIR, 'Mo.upf'))
s_upf = read_upf_header(os.path.join(PSEUDO_DIR, 'S.upf'))

z_valence_Mo = mo_upf['z_valence']  # 14.0 (includes semicore 3d)
z_valence_S = s_upf['z_valence']    # 6.0

# Recommended cutoffs in Hartree (from get_pseudopotential results)
# Mo: 40.0 Ha, S: 26.0 Ha
# INVARIANT 5: 1 Ha = 2 Ry, so convert to Rydberg
ecutwfc_Mo_ry = 2.0 * 40.0  # 80.0 Ry
ecutwfc_S_ry = 2.0 * 26.0   # 52.0 Ry

# Use the maximum cutoff to be safe for both species
ecutwfc = max(ecutwfc_Mo_ry, ecutwfc_S_ry)  # 80.0 Ry
assert ecutwfc == 80.0, f"Expected ecutwfc=80.0 Ry, got {ecutwfc}"

# ecutrho = 4 * ecutwfc (standard for charge density)
ecutrho = 4.0 * ecutwfc  # 320.0 Ry
assert ecutrho == 320.0, f"Expected ecutrho=320.0 Ry, got {ecutrho}"

# Total valence electrons for 6-atom cell (2 Mo + 4 S)
n_electrons = 2 * z_valence_Mo + 4 * z_valence_S  # 2*14 + 4*6 = 52
n_occupied = int(n_electrons / 2)  # 26 occupied bands (spin-unpolarized)

# nbnd for bands: pad well above occupied bands
nbnd = 60
assert nbnd > n_occupied, f"nbnd={nbnd} must be > n_occupied={n_occupied}"

# SCF settings
scf_input_data = {
    'calculation': 'scf',
    'prefix': 'pw',
    'outdir': './tmp',
    'pseudo_dir': '.',
    'ecutwfc': ecutwfc,       # Ry
    'ecutrho': ecutrho,       # Ry
    'occupations': 'fixed',   # semiconductor, gap ~1.46 eV
    'conv_thr': 1.0e-8,       # Ry
    'nspin': 1,               # non-magnetic
}

# Bands settings
# diagonalization='cg' (conjugate gradient) is more robust than Davidson for
# high-lying bands in band structure calculations with many k-points.
bands_input_data = {
    'calculation': 'bands',
    'prefix': 'pw',
    'outdir': './tmp',
    'pseudo_dir': '.',
    'ecutwfc': ecutwfc,       # Ry
    'ecutrho': ecutrho,       # Ry
    'occupations': 'fixed',
    'conv_thr': 1.0e-8,       # Ry
    'nspin': 1,
    'nbnd': nbnd,             # padded above 26 occupied bands
    'diagonalization': 'cg',  # more robust for high-lying bands
}

# Pseudopotential mapping for write_espresso_in
pseudopotentials = {
    'Mo': 'Mo.upf',
    'S': 'S.upf',
}

# Process each structure
for i in range(51):
    nnn = f"{i:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotentials into the calculation directory
    for sp in ['Mo', 'S']:
        src = os.path.join(PSEUDO_DIR, f"{sp}.upf")
        dst = os.path.join(calc_dir, f"{sp}.upf")
        shutil.copy2(src, dst)
    
    # Read the structure
    atoms = ase.io.read(f"structures/structure_{nnn}.cif")
    
    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 6, f"Structure {nnn}: expected 6 atoms, got {nat}"
    assert ntyp == 2, f"Structure {nnn}: expected 2 species, got {ntyp}"
    
    # Get the standard k-path for this structure
    kp = standard_kpath(atoms)
    
    # The primitive cell for the calculation (INVARIANT 3)
    prim = kp_atoms(kp)
    
    # Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))
    
    # SCF k-point mesh (computed, not hand-typed)
    kpts_scf = scf_kmesh(prim, metal=False)
    
    # Write pw.in (SCF)
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=scf_input_data,
        pseudopotentials=pseudopotentials,
        kpts=kpts_scf,
    )
    
    # Write bands.in (NSCF bands)
    kpts_bands = kp_kpts(kp)
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=bands_input_data,
        pseudopotentials=pseudopotentials,
        kpts=kpts_bands,
    )
    
    # Write bands_post.in (bands.x input)
    # &BANDS namelist: prefix, outdir, filband, lsym
    bands_post_content = f"""&BANDS
    prefix = 'pw'
    outdir = './tmp'
    filband = '{nnn}.bands.dat'
    lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)
    
    print(f"Setup complete for calculations/{nnn}/")

print("\nAll 51 calculation directories set up.")
