#!/usr/bin/env python3
"""
setup_jobs.py -- Generate QE input files for all 51 MgO structures.

For each structure NNN (000-050):
  calculations/NNN/
    Mg.pbesol-sr-rrkjus_psl.1.0.0.UPF   (copied from template/)
    O.pbesol-sr-rrkjus_psl.1.0.0.UPF    (copied from template/)
    kpath.json                           (from chem_llm.kpath)
    pw.in                                (SCF input, written by write_espresso_in)
    bands.in                             (bands input, written by write_espresso_in)
    bands_post.in                        (bands.x post-processing input)

Key decisions (see notes):
  - ecutwfc = 92 Ry (46 Ha * 2, from O pseudopotential, INVARIANT 5)
  - ecutrho = 368 Ry (4 * ecutwfc)
  - occupations = 'fixed' (MgO is a clear insulator, gap ~4.4 eV)
  - conv_thr = 1e-8 Ry
  - nspin = 1 (non-magnetic)
  - nbnd = 16 (8 occupied + 8 conduction bands)
  - k-point mesh for SCF: computed via scf_kmesh(prim, metal=False)
  - k-path for bands: from chem_llm.kpath (INVARIANT 3)
"""

import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# Pseudopotential files (copied into each calculations/NNN/)
PP_MG = 'template/Mg.pbesol-sr-rrkjus_psl.1.0.0.UPF'
PP_O = 'template/O.pbesol-sr-rrkjus_psl.1.0.0.UPF'

# Pseudopotential mapping for write_espresso_in
PSEUDOS = {
    'Mg': os.path.basename(PP_MG),
    'O': os.path.basename(PP_O),
}

# Cutoffs (INVARIANT 5: 1 Ha = 2 Ry)
# Mg: 42 Ha = 84 Ry, O: 46 Ha = 92 Ry -> use max = 92 Ry
ECUTWFC_RY = 92.0
ECUTRHO_RY = 4.0 * ECUTWFC_RY  # 368 Ry

# Other settings
PREFIX = 'mgo'
OUTDIR = './tmp'
CONV_THR = 1e-8  # Ry
NBND = 16  # 8 occupied + 8 conduction

N_STRUCTURES = 51

for i in range(N_STRUCTURES):
    nnn = f'{i:03d}'
    calc_dir = f'calculations/{nnn}'
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(os.path.join(calc_dir, 'tmp'), exist_ok=True)

    # Copy pseudopotentials
    shutil.copy(PP_MG, calc_dir)
    shutil.copy(PP_O, calc_dir)

    # Read the structure
    atoms = ase.io.read(f'structures/structure_{nnn}.cif')

    # Get the standard k-path and the primitive cell it refers to (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))

    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(prim)
    ntyp = len(set(prim.get_chemical_symbols()))
    assert nat == 2, f"Structure {nnn}: expected nat=2, got {nat}"
    assert ntyp == 2, f"Structure {nnn}: expected ntyp=2, got {ntyp}"

    # SCF k-point mesh (computed, never typed by hand)
    kpts_scf = scf_kmesh(prim, metal=False)

    # Flat input_data dict for write_espresso_in (INVARIANT 6: let ASE route variables)
    input_data_scf = {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': './',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
        'nspin': 1,
    }

    input_data_bands = {
        'calculation': 'bands',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': './',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
        'nspin': 1,
        'nbnd': NBND,
    }

    # Write pw.in (SCF) using the primitive cell from the k-path
    # write_espresso_in(fd, atoms, input_data, pseudopotentials, kpts, ...)
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data_scf,
        pseudopotentials=PSEUDOS,
        kpts=kpts_scf,
    )

    # Write bands.in using the k-path k-points (INVARIANT 3)
    kpts_bands = kp_kpts(kp)
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data_bands,
        pseudopotentials=PSEUDOS,
        kpts=kpts_bands,
    )

    # Write bands_post.in (bands.x input)
    bands_post_content = f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{nnn}.bands.dat'
    lsym = .false.
/"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)

    if i < 3 or i == N_STRUCTURES - 1:
        print(f"Wrote {calc_dir}/: pw.in, bands.in, bands_post.in, kpath.json")

print(f"\nSet up {N_STRUCTURES} calculation directories in calculations/")
print("Done.")
