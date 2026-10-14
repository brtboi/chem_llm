import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# =============================================================================
# setup_jobs.py
# Creates calculations/NNN/ directories with QE input files for each structure
# =============================================================================

# Pseudopotential files
PSEUDO_DIR = 'template'

# Read z_valence from pseudopotential files (INVARIANT 2)
ti_upf = read_upf_header(os.path.join(PSEUDO_DIR, 'Ti.upf'))
o_upf = read_upf_header(os.path.join(PSEUDO_DIR, 'O.upf'))

z_valence_Ti = ti_upf['z_valence']  # 12.0
z_valence_O = o_upf['z_valence']    # 6.0

# Electron count for primitive cell (2 Ti + 4 O)
n_Ti = 2
n_O = 4
n_electrons = n_Ti * z_valence_Ti + n_O * z_valence_O
n_occupied = int(n_electrons / 2)  # spin-unpolarized

print(f"Electron count: {n_electrons}, occupied bands: {n_occupied}")

# nbnd: pad above occupied bands
nbnd = 48
assert nbnd > n_occupied, f"nbnd={nbnd} must be > n_occupied={n_occupied}"

# Cutoffs (INVARIANT 5: convert from Ha to Ry)
# Ti: 42 Ha, O: 46 Ha -> use max = 46 Ha = 92 Ry
ecutwfc_ha = max(ti_upf.get('ecutwfc_ha', 42.0), o_upf.get('ecutwfc_ha', 46.0))
# Actually, read from the recommended values we recorded:
ecutwfc_ha = 46.0  # max of Ti (42) and O (46)
ecutwfc_ry = 2.0 * ecutwfc_ha  # 92 Ry
ecutrho_ry = 4.0 * ecutwfc_ry  # 368 Ry

print(f"ecutwfc: {ecutwfc_ha} Ha = {ecutwfc_ry} Ry")
print(f"ecutrho: {ecutrho_ry} Ry")

# Common QE settings
common_settings = {
    'ecutwfc': ecutwfc_ry,
    'ecutrho': ecutrho_ry,
    'occupations': 'fixed',  # insulator, gap 1.77 eV
    'conv_thr': 1e-8,
    'nspin': 1,  # non-magnetic
}

# Pseudopotential mapping
pseudopotentials = {
    'Ti': 'Ti.upf',
    'O': 'O.upf',
}

# Process each structure
for i in range(51):
    nnn = f"{i:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotentials into calc_dir (pseudo_dir='.' resolves from there)
    for elem, fname in pseudopotentials.items():
        src = os.path.join(PSEUDO_DIR, fname)
        dst = os.path.join(calc_dir, fname)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    # Read structure
    atoms = ase.io.read(f"structures/structure_{nnn}.cif")
    
    # Get k-path and primitive cell (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)
    
    # Save k-path
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))
    
    # Verify nat/ntyp (INVARIANT 1)
    nat = len(prim)
    ntyp = len(set(prim.get_chemical_symbols()))
    assert nat == 6, f"Structure {nnn}: expected nat=6, got {nat}"
    assert ntyp == 2, f"Structure {nnn}: expected ntyp=2, got {ntyp}"
    
    # SCF k-mesh (computed, not hand-typed)
    kpts_scf = scf_kmesh(prim, metal=False)
    
    # SCF input (pw.in)
    scf_input_data = {
        'calculation': 'scf',
        'prefix': f'tio2_{nnn}',
        'outdir': './tmp',
        'pseudo_dir': '.',
        'nat': nat,
        'ntyp': ntyp,
        'ecutwfc': ecutwfc_ry,
        'ecutrho': ecutrho_ry,
        'occupations': 'fixed',
        'conv_thr': 1e-8,
        'nspin': 1,
    }
    
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=scf_input_data,
        pseudopotentials=pseudopotentials,
        kpts=kpts_scf,
    )
    
    # Bands input (bands.in) - non-self-consistent
    bands_input_data = {
        'calculation': 'bands',
        'prefix': f'tio2_{nnn}',
        'outdir': './tmp',
        'pseudo_dir': '.',
        'nat': nat,
        'ntyp': ntyp,
        'ecutwfc': ecutwfc_ry,
        'ecutrho': ecutrho_ry,
        'occupations': 'fixed',
        'conv_thr': 1e-8,
        'nspin': 1,
        'nbnd': nbnd,
    }
    
    # Bands k-points from k-path
    kpts_bands = kp_kpts(kp)
    
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=bands_input_data,
        pseudopotentials=pseudopotentials,
        kpts=kpts_bands,
    )
    
    # Bands post-processing input (bands_post.in) for bands.x
    bands_post_content = f"""&BANDS
    prefix = 'tio2_{nnn}'
    outdir = './tmp'
    filband = '{nnn}.bands.dat'
    lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)
    
    if i % 10 == 0:
        print(f"Processed structure {nnn}: k-mesh={kpts_scf}, k-path kpts={len(kpts_bands)}")

print("Done: created calculations/000 through calculations/050")
