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
#   - Al.upf (pseudopotential, copied from template/)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input, written with write_espresso_in)
#   - bands.in (NSCF bands input, written with write_espresso_in)
#   - bands_post.in (bands.x input, hand-written)
# =============================================================================

# --- Pseudopotential info (from get_pseudopotential result) ---
# Al.upf: z_valence=3.0, relativistic='scalar', recommended_ecutwfc=20.0 Ha
# INVARIANT 5: 1 Ha = 2 Ry, so ecutwfc = 40.0 Ry
# ecutrho = 4 * ecutwfc = 160.0 Ry (standard 4x ratio for charge density)

# Read z_valence from the actual UPF file (INVARIANT 2)
upf_path = 'template/Al.upf'
upf_header = read_upf_header(upf_path)
z_valence = upf_header['z_valence']
print(f"Al z_valence from UPF: {z_valence}")

# --- QE settings ---
# ecutwfc: 20.0 Ha * 2 = 40.0 Ry (INVARIANT 5)
ecutwfc_ry = 40.0
assert abs(ecutwfc_ry - 2.0 * 20.0) < 1e-10, "ecutwfc conversion error"
# ecutrho: 4x ecutwfc = 160.0 Ry
ecutrho_ry = 4.0 * ecutwfc_ry
assert abs(ecutrho_ry - 160.0) < 1e-10, "ecutrho conversion error"

# Al is a metal (band_gap_eV=0.0), so use smearing
# occupations='smearing', smearing='mv' (Methfessel-Paxton), degauss=0.01 Ry
# conv_thr=1e-8 Ry

# nbnd for bands: Al has z_valence=3.0 per atom, 1 atom -> 3 electrons total.
# For spin-unpolarized, occupied bands = 3/2 = 1.5, so nbnd >= 2.
# For a metal, we need more bands to see conduction bands. Set nbnd=10.
nbnd = 10

# --- Common input_data for SCF and bands ---
# INVARIANT 6: pass a FLAT dict to write_espresso_in; ASE routes each key
# to its correct namelist (conv_thr -> electrons, ecutwfc -> system, etc.)

common_input_data = {
    'prefix': 'al',
    'outdir': './tmp',
    'pseudo_dir': '.',
    'ecutwfc': ecutwfc_ry,   # Ry
    'ecutrho': ecutrho_ry,   # Ry
    'occupations': 'smearing',
    'smearing': 'mv',
    'degauss': 0.01,         # Ry
    'conv_thr': 1e-8,        # Ry
}

# --- Loop over structures ---
for i in range(51):
    nnn = f"{i:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotential into calc_dir (pseudo_dir='.' resolves from here)
    shutil.copy2(upf_path, os.path.join(calc_dir, 'Al.upf'))
    
    # Read the structure
    atoms = ase.io.read(f"structures/structure_{nnn}.cif")
    
    # INVARIANT 1: compute nat/ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 1, f"Structure {nnn}: expected 1 atom, got {nat}"
    assert ntyp == 1, f"Structure {nnn}: expected 1 type, got {ntyp}"
    
    # INVARIANT 3: get the k-path and primitive cell from chem_llm.kpath
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    
    # Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))
    
    # --- SCF k-point mesh (RULE: never typed by hand) ---
    # metal=True because Al is a metal
    kpts_scf = scf_kmesh(prim, metal=True)
    print(f"Structure {nnn}: SCF k-mesh = {kpts_scf}")
    
    # --- Write pw.in (SCF) ---
    scf_input_data = dict(common_input_data)
    scf_input_data['calculation'] = 'scf'
    
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=scf_input_data,
        pseudopotentials={'Al': 'Al.upf'},
        kpts=kpts_scf,
    )
    
    # --- Write bands.in (NSCF bands) ---
    bands_input_data = dict(common_input_data)
    bands_input_data['calculation'] = 'bands'
    bands_input_data['nbnd'] = nbnd
    
    # kpts for bands: the k-path from chem_llm.kpath (INVARIANT 3)
    kpts_bands = kp_kpts(kp)
    
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=bands_input_data,
        pseudopotentials={'Al': 'Al.upf'},
        kpts=kpts_bands,
    )
    
    # --- Write bands_post.in (bands.x input) ---
    # &BANDS namelist: prefix, outdir, filband, lsym
    # lsym=.false. to skip band-representation file (saves time)
    bands_post_content = f"""&BANDS
    prefix = 'al'
    outdir = './tmp'
    filband = '{nnn}.bands.dat'
    lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)
    
    print(f"Structure {nnn}: done")

print("All 51 calculation directories created.")
