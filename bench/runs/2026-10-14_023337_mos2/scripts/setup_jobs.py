"""
setup_jobs.py: For each structure NNN (000-050), create calculations/NNN/ with:
  - Pseudopotential files (Mo.upf, S.upf) copied from template/
  - kpath.json from chem_llm.kpath
  - pw.in (SCF input)
  - bands.in (NSCF bands input)
  - bands_post.in (bands.x post-processing input)

All QE input files are written with ase.io.espresso.write_espresso_in.
"""
import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# Pseudopotential files
PSEUDO_FILES = ['Mo.upf', 'S.upf']
TEMPLATE_DIR = 'template'

# Pseudopotential mapping for write_espresso_in (species -> filename)
# The files are copied into calculations/NNN/ so pseudo_dir='.' resolves them.
PSEUDOS = {
    'Mo': 'Mo.upf',
    'S': 'S.upf',
}

# Cutoffs (INVARIANT 5: convert from Ha to Ry)
# Mo: 40.0 Ha, S: 26.0 Ha -> max = 40.0 Ha = 80.0 Ry
ECUTWFC_RY = 80.0
ECUTRHO_RY = 4 * ECUTWFC_RY  # 320.0 Ry

# NBND: occupied bands = 26 (52 electrons / 2), pad to 40
NBND = 40

# SCF settings
SCF_INPUT = {
    'calculation': 'scf',
    'prefix': 'pwscf',
    'outdir': '.',
    'pseudo_dir': '.',
    'ecutwfc': ECUTWFC_RY,
    'ecutrho': ECUTRHO_RY,
    'occupations': 'fixed',
    'conv_thr': 1e-8,
    'nspin': 1,
}

# Bands settings
BANDS_INPUT = {
    'calculation': 'bands',
    'prefix': 'pwscf',
    'outdir': '.',
    'pseudo_dir': '.',
    'ecutwfc': ECUTWFC_RY,
    'ecutrho': ECUTRHO_RY,
    'occupations': 'fixed',
    'conv_thr': 1e-8,
    'nspin': 1,
    'nbnd': NBND,
}

for i in range(51):
    nnn = f"{i:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotentials
    for pp in PSEUDO_FILES:
        src = os.path.join(TEMPLATE_DIR, pp)
        dst = os.path.join(calc_dir, pp)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    # Read structure
    atoms = ase.io.read(f"structures/structure_{nnn}.cif")
    
    # Get k-path and primitive cell (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)
    
    # Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))
    
    # SCF k-mesh (computed, not hand-typed)
    kpts_scf = scf_kmesh(prim, metal=False)
    
    # Write pw.in (SCF)
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=SCF_INPUT,
        pseudopotentials=PSEUDOS,
        kpts=kpts_scf,
    )
    
    # Write bands.in (NSCF bands)
    kpts_bands = kp_kpts(kp)
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=BANDS_INPUT,
        pseudopotentials=PSEUDOS,
        kpts=kpts_bands,
    )
    
    # Write bands_post.in (bands.x input)
    bands_post = f"""&BANDS
    prefix = 'pwscf'
    outdir = '.'
    filband = '{nnn}.bands.dat'
    lsym = .false.
    spin_component = 1
/"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post)
    
    print(f"Setup complete for {calc_dir}")

print("All 51 calculation directories set up.")
