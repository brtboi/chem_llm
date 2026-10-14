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
#   - pw.in (SCF input, written with write_espresso_in)
#   - bands.in (NSCF bands input, written with write_espresso_in)
#   - bands_post.in (bands.x post-processing input)
#
# QE settings (from STEP 4 note):
#   ecutwfc = 114.0 Ry  (max(57.0, 42.0) Ha * 2.0, INVARIANT 5)
#   ecutrho = 456.0 Ry  (4 * ecutwfc)
#   occupations = 'fixed' (GaN is a wide-gap semiconductor)
#   conv_thr = 1.0e-8 Ry
#   nbnd = 40 (>= 26 occupied bands, with margin)
#   prefix = 'pwscf'
#   outdir = './tmp'
#   pseudo_dir = '.' (pseudos copied into calculations/NNN/)
#   nspin = 1 (no magnetism)
# =============================================================================

# Pseudopotential files (basename only, since they are copied into calc_dir)
PSEUDO_FILES = {
    'Ga': 'Ga.pbesol-sr-rrkjus_psl.1.0.0.UPF',
    'N': 'N.pbesol-sr-rrkjus_psl.1.0.0.UPF',
}

# Cutoffs in Ry (converted from Ha per INVARIANT 5: 1 Ha = 2 Ry)
ECUTWFC_RY = 114.0  # max(57.0, 42.0) * 2.0
ECUTRHO_RY = 456.0  # 4 * ecutwfc

# Verify cutoffs from actual pseudopotential files
for elem, fname in PSEUDO_FILES.items():
    header = read_upf_header(os.path.join('template', fname))
    z_val = header['z_valence']
    print(f"{elem}: z_valence={z_val}")

# Total electrons = 2*Ga + 2*N = 2*21 + 2*5 = 52
# Occupied bands = 52/2 = 26 (spin-unpolarized)
# nbnd = 40 (>= 26, with margin for conduction bands)
NBND = 40

# Bands post-processing settings
PREFIX = 'pwscf'
OUTDIR = './tmp'

# Create calculations directory
os.makedirs('calculations', exist_ok=True)

# Process each structure
for i in range(51):
    nnn = f"{i:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotential files into the calculation directory
    for elem, fname in PSEUDO_FILES.items():
        src = os.path.join('template', fname)
        dst = os.path.join(calc_dir, fname)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    # Read the structure
    atoms = ase.io.read(f"structures/structure_{nnn}.cif")
    
    # Verify nat/ntyp invariant (INVARIANT 1)
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 4, f"Structure {nnn}: expected 4 atoms, got {nat}"
    assert ntyp == 2, f"Structure {nnn}: expected 2 types, got {ntyp}"
    
    # Get the standard k-path for this structure
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in (INVARIANT 3)
    
    # Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))
    
    # SCF k-point mesh (computed, not hand-typed)
    kpts_scf = scf_kmesh(prim, metal=False)
    
    # Bands k-points from the k-path
    kpts_bands = kp_kpts(kp)
    
    # Write pw.in (SCF input)
    input_data_scf = {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': 1.0e-8,
        'nspin': 1,
    }
    with open(os.path.join(calc_dir, 'pw.in'), 'w') as f:
        write_espresso_in(f, prim, input_data=input_data_scf, kpts=kpts_scf, pseudopotentials=PSEUDO_FILES)
    
    # Write bands.in (NSCF bands input)
    input_data_bands = {
        'calculation': 'bands',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'nbnd': NBND,
        'nspin': 1,
    }
    with open(os.path.join(calc_dir, 'bands.in'), 'w') as f:
        write_espresso_in(f, prim, input_data=input_data_bands, kpts=kpts_bands, pseudopotentials=PSEUDO_FILES)
    
    # Write bands_post.in (bands.x post-processing input)
    # &BANDS namelist: prefix, outdir, filband, lsym
    # lsym=.false. to skip symmetry analysis (saves time)
    filband = f"{nnn}.bands.dat"
    bands_post_content = f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{filband}'
    lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)
    
    if i % 10 == 0:
        print(f"Set up calculations/{nnn}/ (nat={nat}, ntyp={ntyp}, kpts_scf={kpts_scf})")

print("Done: set up all 51 calculation directories")
