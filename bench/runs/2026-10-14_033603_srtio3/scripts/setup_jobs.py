import os
import shutil
import numpy as np
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
#   - bands_post.in (bands.x post-processing input, hand-written)
# =============================================================================

# --- Pseudopotential files (from template/) ---
# These are the files that get copied into each calculations/NNN/ directory.
# pseudo_dir='.' in the QE input means QE looks for them in the working dir.
PP_FILES = {
    'Sr': 'template/Sr.pbesol-n-rrkjus_psl.1.0.0.UPF',
    'Ti': 'template/Ti.pbesol-n-rrkjus_psl.1.0.0.UPF',
    'O':  'template/O.pbesol-n-rrkjus_psl.1.0.0.UPF',
}

# --- QE settings (decided in STEP 4 note) ---
# ecutwfc: max recommended cutoff is O at 46 Ha = 92 Ry (INVARIANT 5: 1 Ha = 2 Ry)
# ecutrho: 4 * ecutwfc = 368 Ry (standard 4x ratio for charge density)
# conv_thr: 1.0d-8 Ry (production-quality SCF convergence)
# occupations: 'fixed' (semiconductor, gap ~1.77 eV, not a metal)
# nspin: 1 (non-magnetic, moment 0.0)
# nbnd: total electrons = 10(Sr) + 12(Ti) + 3*6(O) = 40, occupied = 20.
#       Pad to 40 to capture conduction bands up to ~10 eV above VBM.
ECUTWFC_RY = 92.0
ECUTRHO_RY = 368.0
CONV_THR = 1.0e-8
NBND = 40
PREFIX = 'srtio3'
OUTDIR = 'tmp_srtio3'

# Pseudopotential filenames (relative to calculations/NNN/)
PP_NAMES = {
    'Sr': 'Sr.pbesol-n-rrkjus_psl.1.0.0.UPF',
    'Ti': 'Ti.pbesol-n-rrkjus_psl.1.0.0.UPF',
    'O':  'O.pbesol-n-rrkjus_psl.1.0.0.UPF',
}

# --- Process each structure ---
for n in range(51):
    nnn = f'{n:03d}'
    calc_dir = f'calculations/{nnn}'
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotential files into calculations/NNN/
    for elem, src in PP_FILES.items():
        dst = os.path.join(calc_dir, PP_NAMES[elem])
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    # Read the structure
    atoms = ase.io.read(f'structures/structure_{nnn}.cif')
    
    # --- INVARIANT 1: nat and ntyp from the actual Atoms object ---
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 5, f"Structure {nnn}: expected nat=5, got {nat}"
    assert ntyp == 3, f"Structure {nnn}: expected ntyp=3, got {ntyp}"
    
    # --- INVARIANT 3: k-path from chem_llm.kpath for THIS structure ---
    # standard_kpath returns the seekpath standard path for this cell.
    # kp_atoms(kp) is the primitive cell the path refers to -- use it for
    # both pw.in and bands.in (not the CIF's cell, which may differ).
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)
    
    # Save kpath.json (holds labels, tick positions, line segments)
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))
    
    # --- SCF k-point mesh (INVARIANT: never typed by hand) ---
    # scf_kmesh computes an even-spacing MP mesh for the primitive cell.
    # metal=False because SrTiO3 is a semiconductor (gap ~1.77 eV).
    kpts_scf = scf_kmesh(prim, metal=False)
    
    # --- Bands k-points (INVARIANT 3: from kp_kpts(kp)) ---
    kpts_bands = kp_kpts(kp)
    
    # --- Write pw.in (SCF) ---
    # input_data is a FLAT dict; write_espresso_in routes each key to the
    # correct namelist (INVARIANT 6: do NOT hand-sort into &CONTROL/&SYSTEM/etc.)
    scf_input_data = {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
        'nspin': 1,
    }
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=scf_input_data,
        pseudopotentials=PP_NAMES,
        kpts=kpts_scf,
    )
    
    # --- Write bands.in (NSCF bands) ---
    bands_input_data = {
        'calculation': 'bands',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
        'nspin': 1,
        'nbnd': NBND,
    }
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=bands_input_data,
        pseudopotentials=PP_NAMES,
        kpts=kpts_bands,
    )
    
    # --- Write bands_post.in (bands.x input) ---
    # bands.x reads from the outdir written by pw.x (bands calculation).
    # filband is the output file name for the band energies.
    # lsym=.false. skips the band-representation file (saves time).
    # spin_component=1 (majority spin; nspin=1 so only one component).
    bands_post_content = f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{nnn}.bands.dat'
    lsym = .false.
    spin_component = 1
/"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)
    
    if n <= 3 or n % 10 == 0:
        print(f"Setup {nnn}: nat={nat}, ntyp={ntyp}, kpts_scf={kpts_scf}, n_kpts_bands={len(kpts_bands)}")

print("\nDone. All 51 calculation directories set up.")
