#!/usr/bin/env python3
"""
setup_jobs.py -- Generate QE input files for all 51 ZnO structures.

For each structure NNN (000-050):
  calculations/NNN/
    Zn.pbesol-rrkjus_psl.1.0.0.UPF   (pseudopotential, copied from template/)
    O.pbesol-rrkjus_psl.1.0.0.UPF    (pseudopotential, copied from template/)
    kpath.json                        (from chem_llm.kpath)
    pw.in                             (SCF input, written by write_espresso_in)
    bands.in                          (NSCF bands input, written by write_espresso_in)
    bands_post.in                     (bands.x post-processing input)

Key settings (see STEP 4 note):
  ecutwfc  = 92.0 Ry  (46.0 Ha * 2, max of Zn=42 Ha and O=46 Ha)
  ecutrho  = 368 Ry   (4 * ecutwfc)
  conv_thr = 1e-8 Ry  (in &ELECTRONS, not &CONTROL -- INVARIANT 6)
  occupations = 'fixed' (ZnO is a wide-gap semiconductor)
  nspin = 1 (non-magnetic)
  nbnd = 40 (occupied bands = 52/2 = 26, pad above)
  SCF k-mesh: computed via scf_kmesh(prim, metal=False) from chem_llm.kpath
  Bands k-path: kp_kpts(kp) from chem_llm.kpath (INVARIANT 3)
"""

import os
import shutil
import numpy as np
import ase.io
from ase import Atoms
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# --- Pseudopotential files ---
PP_ZN = 'Zn.pbesol-rrkjus_psl.1.0.0.UPF'
PP_O  = 'O.pbesol-rrkjus_psl.1.0.0.UPF'

# --- QE settings ---
ECUTWFC_RY = 92.0   # 46.0 Ha * 2 (INVARIANT 5: 1 Ha = 2 Ry)
ECUTRHO_RY = 368.0  # 4 * ecutwfc
CONV_THR   = 1e-8   # Ry, in &ELECTRONS
NBND       = 40     # occupied = 26, pad above
PREFIX     = 'zno'
OUTDIR     = './tmp'

# Pseudopotential mapping for write_espresso_in
PSEUDOS = {'Zn': PP_ZN, 'O': PP_O}

# --- Read base structure to get the primitive cell ---
# The primitive cell is the one from kp_atoms(standard_kpath(atoms))
# For structure_000 (unperturbed), this is the same as the base cell.
# For perturbed structures, the cell may be slightly different due to strain,
# but the k-path is still computed from the perturbed structure.
base = ase.io.read('structures/structure_000.cif')
print(f"Base structure: {len(base)} atoms, species: {sorted(set(base.get_chemical_symbols()))}")

# --- Create calculations directories and write input files ---
for i in range(51):
    nnn = f"{i:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotentials into the calculation directory
    # (pw.x runs inside calculations/NNN, so pseudo_dir='.' resolves from there)
    for pp in [PP_ZN, PP_O]:
        src = f"template/{pp}"
        dst = f"{calc_dir}/{pp}"
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    # Read the structure
    atoms = ase.io.read(f"structures/structure_{nnn}.cif")
    
    # INVARIANT 1: nat and ntyp computed from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 4, f"Structure {nnn}: nat={nat} != 4"
    assert ntyp == 2, f"Structure {nnn}: ntyp={ntyp} != 2"
    
    # Get the standard k-path for THIS structure (INVARIANT 3)
    # Perturbed structures have lower symmetry and their own path
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    
    # Save kpath.json (holds labels, tick positions, line segments)
    kp_save(kp, f"{calc_dir}/kpath.json")
    
    # --- SCF k-mesh (Monkhorst-Pack, computed from the primitive cell) ---
    # ZnO is a semiconductor (not a metal), so metal=False
    kpts_scf = scf_kmesh(prim, metal=False)
    print(f"  {nnn}: SCF k-mesh = {kpts_scf}")
    
    # --- Write pw.in (SCF) ---
    input_data_scf = {
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
        f"{calc_dir}/pw.in",
        prim,
        input_data=input_data_scf,
        pseudopotentials=PSEUDOS,
        kpts=kpts_scf,
    )
    
    # --- Write bands.in (NSCF bands) ---
    input_data_bands = {
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
        f"{calc_dir}/bands.in",
        prim,
        input_data=input_data_bands,
        pseudopotentials=PSEUDOS,
        kpts=kp_kpts(kp),  # INVARIANT 3: k-points from the k-path
    )
    
    # --- Write bands_post.in (bands.x post-processing) ---
    # &BANDS namelist: prefix, outdir, filband, lsym=.false.
    # lsym=.false. to skip the band-representation file (saves time)
    bands_post_content = f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{nnn}.bands.dat'
    lsym = .false.
/
"""
    with open(f"{calc_dir}/bands_post.in", 'w') as f:
        f.write(bands_post_content)
    
    if i < 3 or i == 50:
        print(f"  Wrote {calc_dir}/: pw.in, bands.in, bands_post.in, kpath.json")

print("\nDone. All 51 calculation directories created.")
