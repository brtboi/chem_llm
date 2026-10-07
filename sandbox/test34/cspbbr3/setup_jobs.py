import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save
from chem_llm.tools.upf import read_upf_header

# =============================================================================
# setup_jobs.py
# Creates calculations/NNN/ directories for each structure with:
#   - pseudopotential files (copied from template/)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input)
#   - bands.in (non-self-consistent bands input)
#   - bands_post.in (bands.x post-processing input)
# =============================================================================

# --- Pseudopotential files ---
# Fetched in STEP 3 into template/. We need to read z_valence from each file
# to compute the total electron count (INVARIANT 2).
pp_files = {
    'Cs': 'template/Cs.pbesol_upf.upf',
    'Pb': 'template/Pb.pbesol_upf.upf',
    'Br': 'template/Br.pbesol_upf.upf',
}

# Read z_valence from each pseudopotential file (INVARIANT 2)
z_valences = {}
for elem, path in pp_files.items():
    header = read_upf_header(path)
    z_valences[elem] = header['z_valence']
    print(f"{elem}: z_valence = {z_valences[elem]}")

# Total electrons = sum of z_valence over all atoms in the structure
# For CsPbBr3: 1 Cs + 1 Pb + 3 Br
total_electrons = z_valences['Cs'] + z_valences['Pb'] + 3 * z_valences['Br']
print(f"Total electrons: {total_electrons}")

# With spin-orbit coupling (noncolin=.true.), each band holds ONE electron,
# so occupied bands = total electrons = 82.
# nbnd must be > 82; we use nbnd = 100 to be safe.
nbnd = 100
assert nbnd > total_electrons, f"nbnd ({nbnd}) must be > total electrons ({total_electrons})"

# --- Cutoffs (INVARIANT 5) ---
# Recommended cutoffs from get_pseudopotential are in HARTREE.
# QE's ecutwfc is in RYDBERG. 1 Ha = 2 Ry.
# Cs: 25 Ha = 50 Ry
# Pb: 44 Ha = 88 Ry (max)
# Br: 38 Ha = 76 Ry
ecutwfc_ry = 2.0 * 44.0  # 88 Ry (from Pb, the max)
ecutrho_ry = 4.0 * ecutwfc_ry  # 352 Ry
print(f"ecutwfc = {ecutwfc_ry} Ry, ecutrho = {ecutrho_ry} Ry")

# --- SCF k-point mesh ---
# 8x8x8 Monkhorst-Pack for a 5-atom primitive cell with ~6 A lattice constant.
scf_kpts = (8, 8, 8)

# --- Common input settings ---
prefix = 'cspbbr3'
outdir = './tmp'
pseudo_dir = '.'  # pseudopotentials will be copied into calculations/NNN/

# =============================================================================
# Loop over all 51 structures
# =============================================================================
for i in range(51):
    nnn = f"{i:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotential files into the calculation directory
    for elem, pp_path in pp_files.items():
        shutil.copy2(pp_path, os.path.join(calc_dir, os.path.basename(pp_path)))
    
    # Read the structure
    atoms = ase.io.read(f"structures/structure_{nnn}.cif")
    
    # INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 5, f"Structure {nnn}: expected 5 atoms, got {nat}"
    assert ntyp == 3, f"Structure {nnn}: expected 3 types, got {ntyp}"
    
    # Get the k-path for this structure (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    
    # Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))
    
    # --- Write pw.in (SCF) ---
    # Flat input_data dict; ASE routes each key to the correct namelist (INVARIANT 6)
    scf_input_data = {
        'calculation': 'scf',
        'prefix': prefix,
        'outdir': outdir,
        'pseudo_dir': pseudo_dir,
        'ecutwfc': ecutwfc_ry,
        'ecutrho': ecutrho_ry,
        'occupations': 'fixed',  # semiconductor, band gap ~1.78 eV
        'conv_thr': 1.0e-8,  # Ry
        'noncolin': True,  # spin-orbit coupling
        'lspinorb': True,  # use spin-orbit pseudopotentials
    }
    
    # Pseudopotential filenames (relative to pseudo_dir='.')
    pseudos = {
        'Cs': 'Cs.pbesol_upf.upf',
        'Pb': 'Pb.pbesol_upf.upf',
        'Br': 'Br.pbesol_upf.upf',
    }
    
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=scf_input_data,
        pseudopotentials=pseudos,
        kpts=scf_kpts,
    )
    
    # --- Write bands.in (non-self-consistent bands) ---
    bands_input_data = {
        'calculation': 'bands',
        'prefix': prefix,
        'outdir': outdir,
        'pseudo_dir': pseudo_dir,
        'ecutwfc': ecutwfc_ry,
        'ecutrho': ecutrho_ry,
        'nbnd': nbnd,
        'noncolin': True,
        'lspinorb': True,
    }
    
    # kpts from the k-path (INVARIANT 3)
    bands_kpts = kp_kpts(kp)
    
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=bands_input_data,
        pseudopotentials=pseudos,
        kpts=bands_kpts,
    )
    
    # --- Write bands_post.in (bands.x input) ---
    # bands.x uses a simple namelist &BANDS with prefix, outdir, filband.
    # noncolin and lspinorb are NOT valid variables in &BANDS -- they belong
    # to &SYSTEM in pw.x. bands.x reads the wavefunctions from pw.x output,
    # which already has the spin-orbit information encoded.
    bands_post_content = f"""&BANDS
    prefix = '{prefix}'
    outdir = '{outdir}'
    filband = '{nnn}.bands.dat'
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)
    
    if i % 10 == 0:
        print(f"  Set up {calc_dir}/ (nat={nat}, ntyp={ntyp})")

print(f"\nDone. Set up 51 calculation directories in calculations/.")
print(f"nbnd = {nbnd}, total electrons = {total_electrons}")
print(f"ecutwfc = {ecutwfc_ry} Ry, ecutrho = {ecutrho_ry} Ry")
print(f"SCF k-mesh: {scf_kpts}")
