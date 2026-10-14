import os
import shutil
import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from ase.io.espresso import write_espresso_in

# =============================================================================
# setup_jobs.py
# For each structure NNN, create calculations/NNN/ with:
#   - pseudopotential files (Pb.upf, Te.upf)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input for pw.x)
#   - bands.in (NSCF bands input for pw.x)
#   - bands_post.in (bands.x post-processing input)
# =============================================================================

# --- Pseudopotential settings (from STEP 3) ---
# Pb: z_valence=22, fully relativistic, spin-orbit, recommended ecutwfc=44.0 Ha
# Te: z_valence=24, fully relativistic, spin-orbit, recommended ecutwfc=51.0 Ha
# INVARIANT 5: convert Ha to Ry (1 Ha = 2 Ry)
# Use max recommended cutoff: 51.0 Ha -> 102.0 Ry
ecutwfc_ha = 51.0  # Ha (Te's recommended cutoff, the higher of the two)
ecutwfc_ry = 2.0 * ecutwfc_ha  # = 102.0 Ry
ecutrho_ry = 4.0 * ecutwfc_ry  # = 408.0 Ry (standard 4x ratio)
assert ecutwfc_ry == 102.0, f"Expected ecutwfc=102.0 Ry, got {ecutwfc_ry}"
assert ecutrho_ry == 408.0, f"Expected ecutrho=408.0 Ry, got {ecutrho_ry}"

# --- Electron count (INVARIANT 2) ---
# Pb z_valence=22, Te z_valence=24 -> total 46 electrons per formula unit
# With spin-orbit (noncolin), each band holds 1 electron
# So occupied bands = 46 (not 23)
# nbnd must be >= 46 + conduction bands. Set nbnd=64 (46 occ + 18 cond).
n_electrons = 22 + 24  # = 46
n_occupied = n_electrons  # spin-orbit: 1 electron per band
nbnd = 64
assert nbnd >= n_occupied, f"nbnd={nbnd} must be >= n_occupied={n_occupied}"

# --- Common settings ---
prefix = 'pbte'
outdir = './tmp'
pseudo_dir = '.'  # pseudopotentials copied into calculations/NNN/
conv_thr = 1.0e-8  # Ry

# Spin-orbit: noncolin=.true., lspinorb=.true. in &SYSTEM
# Pb (Z=82) > 54, so fully relativistic pseudopotentials with spin-orbit.

# --- Pseudopotential filenames ---
pseudopotentials = {
    'Pb': 'Pb.upf',
    'Te': 'Te.upf',
}

# --- Structures to process ---
# The task says to run DFT only for calculations/000, but setup_jobs.py
# should generate inputs for all 51 structures. We'll generate all, but
# only run 000 in STEP 6.
num_structures = 51

for nnn in range(num_structures):
    nnn_str = f"{nnn:03d}"
    calc_dir = f"calculations/{nnn_str}"
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotentials into the calculation directory
    for elem, fname in pseudopotentials.items():
        src = f"template/{fname}"
        dst = f"{calc_dir}/{fname}"
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    # Read the structure
    cif_path = f"structures/structure_{nnn_str}.cif"
    atoms = ase.io.read(cif_path)
    
    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Structure {nnn_str}: expected nat=2, got {nat}"
    assert ntyp == 2, f"Structure {nnn_str}: expected ntyp=2, got {ntyp}"
    
    # INVARIANT 3: k-path from chem_llm.kpath for THIS structure
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    
    # Save kpath.json
    kp_save(kp, f"{calc_dir}/kpath.json")
    
    # SCF k-point mesh (Monkhorst-Pack, semiconductor -> metal=False)
    kpts_scf = scf_kmesh(prim, metal=False)
    
    # --- SCF input (pw.in) ---
    input_data_scf = {
        'control': {
            'calculation': 'scf',
            'prefix': prefix,
            'outdir': outdir,
            'pseudo_dir': pseudo_dir,
        },
        'system': {
            'ecutwfc': ecutwfc_ry,
            'ecutrho': ecutrho_ry,
            'noncolin': True,
            'lspinorb': True,
            'occupations': 'fixed',
        },
        'electrons': {
            'conv_thr': conv_thr,
        },
    }
    
    write_espresso_in(
        f"{calc_dir}/pw.in",
        prim,
        input_data=input_data_scf,
        pseudopotentials=pseudopotentials,
        kpts=kpts_scf,
    )
    
    # --- Bands input (bands.in) ---
    # calculation='bands' for NSCF band structure
    # kpts = the k-path points from chem_llm.kpath
    kpts_bands = kp_kpts(kp)
    
    input_data_bands = {
        'control': {
            'calculation': 'bands',
            'prefix': prefix,
            'outdir': outdir,
            'pseudo_dir': pseudo_dir,
        },
        'system': {
            'ecutwfc': ecutwfc_ry,
            'ecutrho': ecutrho_ry,
            'noncolin': True,
            'lspinorb': True,
            'occupations': 'fixed',
            'nbnd': nbnd,
        },
        'electrons': {
            'conv_thr': conv_thr,
        },
    }
    
    write_espresso_in(
        f"{calc_dir}/bands.in",
        prim,
        input_data=input_data_bands,
        pseudopotentials=pseudopotentials,
        kpts=kpts_bands,
    )
    
    # --- Bands post-processing input (bands_post.in) ---
    # bands.x input: &BANDS namelist
    # filband = 'NNN.bands.dat' (the output file from bands.x)
    # lsym = .false. (skip band-representation file, saves time with spin-orbit)
    bands_post_content = f"""&BANDS
    prefix = '{prefix}'
    outdir = '{outdir}'
    filband = '{nnn_str}.bands.dat'
    lsym = .false.
/
"""
    with open(f"{calc_dir}/bands_post.in", 'w') as f:
        f.write(bands_post_content)
    
    if nnn < 3 or nnn == num_structures - 1:
        print(f"Generated {calc_dir}/: pw.in, bands.in, bands_post.in, kpath.json")
        print(f"  nat={nat}, ntyp={ntyp}, kpts_scf={kpts_scf}, n_kpts_bands={len(kpts_bands)}")

print(f"\nDone. Generated inputs for {num_structures} structures.")
print(f"ecutwfc={ecutwfc_ry} Ry, ecutrho={ecutrho_ry} Ry, nbnd={nbnd}")
print(f"Spin-orbit: noncolin=True, lspinorb=True")
