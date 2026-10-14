import os
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# ============================================================
# STEP 5: Set up QE input files for all 51 structures
# ============================================================
# Pseudopotential info (from get_pseudopotential results):
#   Na: z_valence=9.0, recommended ecutwfc=44.0 Ha
#   Cl: z_valence=7.0, recommended ecutwfc=29.0 Ha
# INVARIANT 5: convert Ha to Ry (1 Ha = 2 Ry)
#   ecutwfc = 2 * max(44.0, 29.0) = 88.0 Ry
#   ecutrho = 4 * ecutwfc = 352.0 Ry
# Total valence electrons per formula unit = 9 + 7 = 16
# Occupied bands = 16/2 = 8 (spin-unpolarized)
# nbnd = 32 (well above 8, room for conduction bands)

# Read pseudopotential headers to get z_valence (INVARIANT 2)
na_pp = read_upf_header('template/Na.upf')
cl_pp = read_upf_header('template/Cl.upf')
print(f"Na z_valence: {na_pp['z_valence']}")
print(f"Cl z_valence: {cl_pp['z_valence']}")

# INVARIANT 5: convert cutoffs from Ha to Ry
ecutwfc_ha = max(na_pp['z_valence'] and 44.0, cl_pp['z_valence'] and 29.0)  # max of recommended cutoffs
# Actually, we need the recommended cutoffs, not z_valence. Let me hardcode from the get_pseudopotential results:
# Na recommended ecutwfc = 44.0 Ha, Cl recommended ecutwfc = 29.0 Ha
ecutwfc_ha = 44.0  # max of Na (44.0) and Cl (29.0) recommended cutoffs in Ha
ecutwfc_ry = 2.0 * ecutwfc_ha  # convert to Ry
ecutrho_ry = 4.0 * ecutwfc_ry  # standard 4x ratio

# INVARIANT 5 check: assert the converted values are what we expect
assert abs(ecutwfc_ry - 88.0) < 1e-10, f"ecutwfc_ry = {ecutwfc_ry}, expected 88.0"
assert abs(ecutrho_ry - 352.0) < 1e-10, f"ecutrho_ry = {ecutrho_ry}, expected 352.0"

# Total valence electrons (from pseudopotential z_valence values)
# For a 2-atom cell (1 Na + 1 Cl): total = 9 + 7 = 16
# Occupied bands = 16/2 = 8
n_electrons = int(na_pp['z_valence'] + cl_pp['z_valence'])
n_occupied = n_electrons // 2
nbnd = 32  # well above n_occupied=8

print(f"Total valence electrons: {n_electrons}")
print(f"Occupied bands: {n_occupied}")
print(f"nbnd: {nbnd}")

# Pseudopotential filenames for write_espresso_in
pseudopotentials = {'Na': 'Na.upf', 'Cl': 'Cl.upf'}

# Common input_data for SCF and bands
# INVARIANT 6: pass a FLAT dict to write_espresso_in; ASE routes each key to the correct namelist
common_input_data = {
    'prefix': 'nacl',
    'outdir': './tmp',
    'pseudo_dir': '../../template',  # relative to calculations/NNN
    'ecutwfc': ecutwfc_ry,           # Ry
    'ecutrho': ecutrho_ry,           # Ry
    'occupations': 'fixed',          # NaCl is an insulator (gap ~5 eV)
    'conv_thr': 1e-8,                # Ry
    'nspin': 1,                      # non-magnetic
}

# SCF-specific
scf_input_data = dict(common_input_data)
scf_input_data['calculation'] = 'scf'

# Bands-specific
bands_input_data = dict(common_input_data)
bands_input_data['calculation'] = 'bands'
bands_input_data['nbnd'] = nbnd

# bands_post.in (bands.x input)
# &BANDS: prefix, outdir, filband, lsym
bands_post_content = """&BANDS
  prefix = 'nacl'
  outdir = './tmp'
  filband = '000.bands.dat'
  lsym = .false.
/
"""

# Process each structure
for i in range(51):
    nnn = f"{i:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(f"{calc_dir}/tmp", exist_ok=True)
    
    # Read the structure
    atoms = ase.io.read(f"structures/structure_{nnn}.cif")
    
    # INVARIANT 1: compute nat/ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Structure {nnn}: expected 2 atoms, got {nat}"
    assert ntyp == 2, f"Structure {nnn}: expected 2 species, got {ntyp}"
    
    # Get the k-path and primitive cell (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    
    # Save kpath.json
    kp_save(kp, f"{calc_dir}/kpath.json")
    
    # SCF k-point mesh (RULE: use scf_kmesh, not hand-typed)
    # NaCl is an insulator, so metal=False
    kpts_scf = scf_kmesh(prim, metal=False)
    print(f"Structure {nnn}: SCF k-mesh = {kpts_scf}")
    
    # Write pw.in (SCF)
    write_espresso_in(
        f"{calc_dir}/pw.in",
        prim,
        input_data=scf_input_data,
        pseudopotentials=pseudopotentials,
        kpts=kpts_scf,
    )
    
    # Write bands.in (NSCF bands)
    # INVARIANT 3: use kp_kpts(kp) for the k-path
    kpts_bands = kp_kpts(kp)
    write_espresso_in(
        f"{calc_dir}/bands.in",
        prim,
        input_data=bands_input_data,
        pseudopotentials=pseudopotentials,
        kpts=kpts_bands,
    )
    
    # Write bands_post.in (bands.x input)
    # filband must be 'NNN.bands.dat'
    bands_post = bands_post_content.replace('000.bands.dat', f'{nnn}.bands.dat')
    with open(f"{calc_dir}/bands_post.in", 'w') as f:
        f.write(bands_post)
    
    print(f"Wrote {calc_dir}/pw.in, {calc_dir}/bands.in, {calc_dir}/bands_post.in")

print("\nDone setting up all 51 calculation directories.")
