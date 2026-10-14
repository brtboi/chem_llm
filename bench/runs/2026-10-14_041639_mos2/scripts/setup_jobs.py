"""
setup_jobs.py - Generate QE input files for all 51 MoS2 structures.

For each structure NNN (000-050):
  - Create calculations/NNN/
  - Copy pseudopotential files (Mo.upf, S.upf) into calculations/NNN/
  - Compute k-path via chem_llm.kpath (INVARIANT 3)
  - Write pw.in (SCF) and bands.in (NSCF bands) with write_espresso_in
  - Write bands_post.in (bands.x input)

Key settings (from STEP 4 note):
  - ecutwfc = 80 Ry (40 Ha * 2, from Mo pseudopotential, INVARIANT 5)
  - ecutrho = 320 Ry (4 * ecutwfc)
  - occupations = 'fixed' (semiconductor, gap ~1.46 eV)
  - conv_thr = 1e-8 Ry
  - nbnd = 64 (56 electrons / 2 = 28 occupied bands, 64 > 28)
  - SCF k-mesh from scf_kmesh(prim, metal=False)
  - Bands kpts from kp_kpts(kp)
"""

import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# Pseudopotential files
PSEUDO_DIR = 'template'
MO_PSEUDO = os.path.join(PSEUDO_DIR, 'Mo.upf')
S_PSEUDO = os.path.join(PSEUDO_DIR, 'S.upf')

# Read z_valence from pseudopotential files (INVARIANT 2)
mo_zval = read_upf_header(MO_PSEUDO)['z_valence']
s_zval = read_upf_header(S_PSEUDO)['z_valence']
print(f"Mo z_valence = {mo_zval}, S z_valence = {s_zval}")

# Cutoffs (INVARIANT 5: convert Ha to Ry)
# Mo: 40 Ha = 80 Ry, S: 26 Ha = 52 Ry
# Use max = 80 Ry for ecutwfc
ECUTWFC_RY = 80.0
ECUTRHO_RY = 4.0 * ECUTWFC_RY  # 320 Ry

# Verify cutoffs from pseudopotential files
mo_ecut_ha = 40.0  # from get_pseudopotential result
s_ecut_ha = 26.0   # from get_pseudopotential result
assert ECUTWFC_RY == 2.0 * mo_ecut_ha, f"ecutwfc mismatch: {ECUTWFC_RY} != 2*{mo_ecut_ha}"

# Number of structures
N_STRUCTURES = 51

# Pseudopotential mapping for write_espresso_in
PSEUDOS = {'Mo': 'Mo.upf', 'S': 'S.upf'}

for nnn in range(N_STRUCTURES):
    calc_dir = f'calculations/{nnn:03d}'
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotential files into calc_dir (pseudo_dir='.' in QE input)
    for pp_file in [MO_PSEUDO, S_PSEUDO]:
        shutil.copy2(pp_file, calc_dir)
    
    # Read structure
    cif_path = f'structures/structure_{nnn:03d}.cif'
    atoms = ase.io.read(cif_path)
    
    # INVARIANT 1: compute nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 6, f"Structure {nnn:03d}: expected nat=6, got {nat}"
    assert ntyp == 2, f"Structure {nnn:03d}: expected ntyp=2, got {ntyp}"
    
    # INVARIANT 3: k-path from chem_llm.kpath for THIS structure
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    
    # Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))
    
    # SCF k-mesh (monkhorst-pack, metal=False for semiconductor)
    kpts_scf = scf_kmesh(prim, metal=False)
    print(f"Structure {nnn:03d}: SCF k-mesh = {kpts_scf}")
    
    # Bands k-points from k-path
    kpts_bands = kp_kpts(kp)
    
    # Total electrons (INVARIANT 2: from pseudopotential z_valence, not textbook)
    symbols = atoms.get_chemical_symbols()
    n_electrons = sum(mo_zval if s == 'Mo' else s_zval for s in symbols)
    n_occupied = int(n_electrons / 2)  # spin-unpolarized
    nbnd = 64  # well above n_occupied=28
    assert nbnd > n_occupied, f"nbnd={nbnd} must be > n_occupied={n_occupied}"
    
    # SCF input (pw.in)
    scf_input_data = {
        'calculation': 'scf',
        'prefix': 'scf',
        'outdir': 'tmp_scf',
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': 1e-8,
    }
    
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=scf_input_data,
        pseudopotentials=PSEUDOS,
        kpts=kpts_scf,
    )
    
    # Bands input (bands.in) - NSCF bands calculation
    bands_input_data = {
        'calculation': 'bands',
        'prefix': 'scf',
        'outdir': 'tmp_scf',
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': 1e-8,
        'nbnd': nbnd,
    }
    
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=bands_input_data,
        pseudopotentials=PSEUDOS,
        kpts=kpts_bands,
    )
    
    # Bands post-processing input (bands_post.in) for bands.x
    bands_post_content = f"""&BANDS
    prefix = 'scf'
    outdir = 'tmp_scf'
    filband = '{nnn:03d}.bands.dat'
    lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)
    
    print(f"Structure {nnn:03d}: done (nat={nat}, ntyp={ntyp}, n_electrons={n_electrons}, n_occupied={n_occupied})")

print("\nAll 51 calculation directories created.")
print("Next: validate calculations/000/pw.in and calculations/000/bands.in")
