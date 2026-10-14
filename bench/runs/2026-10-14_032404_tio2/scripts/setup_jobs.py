import os
import shutil
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# =============================================================================
# setup_jobs.py
# For each structure NNN (000-050), create calculations/NNN/ containing:
#   - pseudopotential files (copied from template/)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input, written with write_espresso_in)
#   - bands.in (NSCF bands input, written with write_espresso_in)
#   - bands_post.in (bands.x post-processing input)
# =============================================================================

# Pseudopotential files (from template/)
# Ti: z_valence=12 (semicore 3s2 3p6 included), O: z_valence=6
# Total electrons = 2*12 + 4*6 = 48, occupied bands = 24 (spin-unpolarized)
# nbnd = 48 (2x occupied, safe padding)

# Cutoffs (INVARIANT 5: 1 Ha = 2 Ry)
# Ti hint: 42 Ha = 84 Ry
# O hint: 46 Ha = 92 Ry
# Use max = 92 Ry for ecutwfc
ECUTWFC_RY = 92.0
ECUTRHO_RY = 4.0 * ECUTWFC_RY  # 368 Ry

# Pseudopotential filenames
PSEUDO_FILES = {
    'Ti': 'Ti.pbesol-sr-rrkjus_psl.1.0.0.UPF',
    'O': 'O.pbesol-sr-rrkjus_psl.1.0.0.UPF',
}

N_STRUCTURES = 51

for i in range(N_STRUCTURES):
    nnn = f"{i:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotential files into calc_dir (pseudo_dir='.' from calc_dir)
    for elem, fname in PSEUDO_FILES.items():
        src = f"template/{fname}"
        dst = f"{calc_dir}/{fname}"
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    # Read the structure
    atoms = ase.io.read(f"structures/structure_{nnn}.cif")
    
    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 6, f"Structure {nnn}: expected 6 atoms, got {nat}"
    assert ntyp == 2, f"Structure {nnn}: expected 2 species, got {ntyp}"
    
    # INVARIANT 3: k-path from chem_llm.kpath for THIS structure
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    
    # Save kpath.json
    kp_save(kp, f"{calc_dir}/kpath.json")
    
    # SCF k-point mesh (Monkhorst-Pack, computed from the cell)
    kpts_scf = scf_kmesh(prim, metal=False)
    
    # Bands k-points (from the k-path)
    kpts_bands = kp_kpts(kp)
    
    # Pseudopotential mapping for write_espresso_in
    pseudos = {elem: fname for elem, fname in PSEUDO_FILES.items()}
    
    # --- SCF input (pw.in) ---
    scf_input_data = {
        'calculation': 'scf',
        'prefix': 'tio2',
        'outdir': './tmp',
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,      # Ry
        'ecutrho': ECUTRHO_RY,      # Ry
        'occupations': 'fixed',     # clear semiconductor
        'conv_thr': 1e-8,           # Ry (in &ELECTRONS, not &CONTROL)
        'nat': nat,                 # from actual Atoms object (INVARIANT 1)
        'ntyp': ntyp,               # from actual Atoms object (INVARIANT 1)
    }
    
    with open(f"{calc_dir}/pw.in", 'w') as f:
        write_espresso_in(f, prim, input_data=scf_input_data,
                          pseudopotentials=pseudos, kpts=kpts_scf)
    
    # --- Bands input (bands.in) ---
    bands_input_data = {
        'calculation': 'bands',
        'prefix': 'tio2',
        'outdir': './tmp',
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,      # Ry
        'ecutrho': ECUTRHO_RY,      # Ry
        'occupations': 'fixed',
        'conv_thr': 1e-8,           # Ry
        'nbnd': 48,                 # 2x occupied bands (24), safe padding
        'nat': nat,
        'ntyp': ntyp,
    }
    
    with open(f"{calc_dir}/bands.in", 'w') as f:
        write_espresso_in(f, prim, input_data=bands_input_data,
                          pseudopotentials=pseudos, kpts=kpts_bands)
    
    # --- Bands post-processing input (bands_post.in) ---
    # bands.x input: &BANDS namelist
    bands_post_content = f"""&BANDS
    prefix = 'tio2'
    outdir = './tmp'
    filband = '{nnn}.bands.dat'
    lsym = .false.
/
"""
    with open(f"{calc_dir}/bands_post.in", 'w') as f:
        f.write(bands_post_content)
    
    print(f"{nnn}: nat={nat}, ntyp={ntyp}, kpts_scf={kpts_scf}, n_kpts_bands={len(kpts_bands)}")

print(f"\nDone: set up {N_STRUCTURES} calculation directories")
