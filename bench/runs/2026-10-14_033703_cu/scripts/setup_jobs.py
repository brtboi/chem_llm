#!/usr/bin/env python3
"""
setup_jobs.py -- Generate QE input files for all 51 Cu structures.

For each structure NNN (000-050):
  calculations/NNN/
    Cu.pbesol_upf   -- pseudopotential (copied from template/)
    kpath.json      -- k-path from chem_llm.kpath
    pw.in           -- SCF input (write_espresso_in)
    bands.in        -- bands input (write_espresso_in)
    bands_post.in   -- bands.x post-processing input

Key decisions (see STEP 4 note):
  - ecutwfc = 104.0 Ry (52.0 Ha * 2, INVARIANT 5)
  - ecutrho = 416.0 Ry (4 * ecutwfc)
  - occupations = 'smearing', smearing = 'mv', degauss = 0.01 Ry (Cu is a metal)
  - conv_thr = 1.0e-8 (in &ELECTRONS, INVARIANT 6)
  - nbnd = 20 (z_valence=19, occupied bands=10, pad to 20)
  - nspin = 1 (non-magnetic)
  - SCF k-mesh: scf_kmesh(prim, metal=True) from chem_llm.kpath
  - bands.in kpts: kp_kpts(kp) (INVARIANT 3)
  - bands_post.in: lsym=.false. (skip band-representation file)
"""

import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# --- Constants ---
N_STRUCTURES = 51
ECUTWFC_RY = 104.0   # 52.0 Ha * 2 (INVARIANT 5)
ECUTRHO_RY = 416.0   # 4 * ecutwfc
NBND = 20            # z_valence=19, occupied=10, pad to 20
CONV_THR = 1.0e-8
DEGAUSS = 0.01       # Ry
PREFIX = 'cu'
OUTDIR = './tmp'

# Pseudopotential file (relative to calculations/NNN/)
PP_FILE = 'Cu.pbesol_upf'
PP_TEMPLATE = 'template/Cu.pbesol_upf'

# --- Build input_data dicts (flat, ASE routes to correct namelists) ---
# SCF input_data
scf_input_data = {
    'calculation': 'scf',
    'prefix': PREFIX,
    'outdir': OUTDIR,
    'pseudo_dir': '.',
    'ecutwfc': ECUTWFC_RY,
    'ecutrho': ECUTRHO_RY,
    'occupations': 'smearing',
    'smearing': 'mv',
    'degauss': DEGAUSS,
    'conv_thr': CONV_THR,
    'nspin': 1,
}

# Bands input_data (calculation='bands', no smearing needed for NSCF)
bands_input_data = {
    'calculation': 'bands',
    'prefix': PREFIX,
    'outdir': OUTDIR,
    'pseudo_dir': '.',
    'ecutwfc': ECUTWFC_RY,
    'ecutrho': ECUTRHO_RY,
    'nspin': 1,
    'nbnd': NBND,
}

# --- Process each structure ---
for nnn in range(N_STRUCTURES):
    calc_dir = f'calculations/{nnn:03d}'
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotential into calc_dir (pseudo_dir='.' resolves from here)
    pp_dest = os.path.join(calc_dir, PP_FILE)
    if not os.path.exists(pp_dest):
        shutil.copy2(PP_TEMPLATE, pp_dest)
    
    # Read structure
    atoms = ase.io.read(f'structures/structure_{nnn:03d}.cif')
    
    # INVARIANT 1: nat/ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 1, f"Structure {nnn:03d}: expected nat=1, got {nat}"
    assert ntyp == 1, f"Structure {nnn:03d}: expected ntyp=1, got {ntyp}"
    
    # Get k-path and primitive cell (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)
    
    # Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))
    
    # SCF k-mesh (metal=True for Cu)
    kpts_scf = scf_kmesh(prim, metal=True)
    
    # Write pw.in (SCF)
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=scf_input_data,
        pseudopotentials={'Cu': PP_FILE},
        kpts=kpts_scf,
    )
    
    # Write bands.in (NSCF bands)
    kpts_bands = kp_kpts(kp)
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=bands_input_data,
        pseudopotentials={'Cu': PP_FILE},
        kpts=kpts_bands,
    )
    
    # Write bands_post.in (bands.x input)
    bands_post_content = f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{nnn:03d}.bands.dat'
    lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)
    
    if nnn <= 2 or nnn % 10 == 0:
        print(f"{calc_dir}: nat={nat}, ntyp={ntyp}, kpts_scf={kpts_scf}, "
              f"n_kpts_bands={len(kpts_bands)}")

print(f"\nDone. Set up {N_STRUCTURES} calculation directories.")
print(f"ecutwfc={ECUTWFC_RY} Ry, ecutrho={ECUTRHO_RY} Ry, nbnd={NBND}")
print(f"occupations=smearing, smearing=mv, degauss={DEGAUSS} Ry")
print(f"conv_thr={CONV_THR}")
