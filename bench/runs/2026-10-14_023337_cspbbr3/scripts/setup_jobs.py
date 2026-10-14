#!/usr/bin/env python3
"""
setup_jobs.py -- Generate QE input files for all 51 CsPbBr3 structures.

For each structure NNN (000-050), creates calculations/NNN/ containing:
  - Cs.upf, Pb.upf, Br.upf (copied from template/)
  - kpath.json (from chem_llm.kpath)
  - pw.in (SCF calculation)
  - bands.in (NSCF bands calculation)
  - bands_post.in (bands.x post-processing input)

Key decisions (see STEP 4 note):
  - ecutwfc = 88 Ry (max of Cs=25, Pb=44, Br=38 Ha, converted: 44 Ha * 2 = 88 Ry)
  - ecutrho = 352 Ry (4 * ecutwfc)
  - noncolin=.true., lspinorb=.true. (spin-orbit, Pb Z=82, Cs Z=55 both >54)
  - nbnd = 100 (82 electrons with spin-orbit, 1 electron/band, padded above 82)
  - occupations='fixed' (semiconductor, gap ~1.78 eV)
  - conv_thr = 1e-8
  - SCF k-mesh: scf_kmesh(prim, metal=False) from chem_llm.kpath
  - bands.in: kpts=kp_kpts(kp) from chem_llm.kpath
  - bands_post.in: &BANDS prefix, outdir, filband='NNN.bands.dat', lsym=.false.
  - pseudo_dir = '.' (pseudopotentials are copied into calculations/NNN/)
"""

import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# Pseudopotential files
PSEUDO_FILES = {
    'Cs': 'Cs.upf',
    'Pb': 'Pb.upf',
    'Br': 'Br.upf',
}

# Cutoffs (converted from Ha to Ry: 1 Ha = 2 Ry)
# Cs: 25 Ha -> 50 Ry, Pb: 44 Ha -> 88 Ry, Br: 38 Ha -> 76 Ry
# Use max = 88 Ry for ecutwfc
ECUTWFC_RY = 88.0
ECUTRHO_RY = 4.0 * ECUTWFC_RY  # 352 Ry

# Spin-orbit: each band holds 1 electron (noncolin)
# Total electrons = 9 (Cs) + 22 (Pb) + 3*17 (Br) = 82
# nbnd must be >= 82; use 100 for safety
NBND = 100

# Common input parameters (flat dict, routed by write_espresso_in)
# pseudo_dir = '.' because pseudopotentials are copied into calculations/NNN/
COMMON_INPUT = {
    'prefix': 'cspbbr3',
    'outdir': './tmp',
    'pseudo_dir': '.',
    'ecutwfc': ECUTWFC_RY,
    'ecutrho': ECUTRHO_RY,
    'occupations': 'fixed',
    'conv_thr': 1e-8,
    'noncolin': True,
    'lspinorb': True,
    'nbnd': NBND,
}


def setup_structure(nnn: int):
    """Set up calculations/NNN/ for structure NNN."""
    calc_dir = f'calculations/{nnn:03d}'
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(f'{calc_dir}/tmp', exist_ok=True)

    # Copy pseudopotential files into the calculation directory
    for elem, fname in PSEUDO_FILES.items():
        src = f'template/{fname}'
        dst = f'{calc_dir}/{fname}'
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Read the structure
    atoms = ase.io.read(f'structures/structure_{nnn:03d}.cif')

    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 5, f"Structure {nnn:03d}: expected 5 atoms, got {nat}"
    assert ntyp == 3, f"Structure {nnn:03d}: expected 3 species, got {ntyp}"

    # Get the standard k-path for THIS structure
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in

    # Save kpath.json
    kp_save(kp, f'{calc_dir}/kpath.json')

    # SCF k-mesh (Monkhorst-Pack, computed for the primitive cell)
    kpts_scf = scf_kmesh(prim, metal=False)

    # Pseudopotential filenames for write_espresso_in
    pseudos = {elem: fname for elem, fname in PSEUDO_FILES.items()}

    # --- Write pw.in (SCF) ---
    input_scf = dict(COMMON_INPUT)
    input_scf['calculation'] = 'scf'
    write_espresso_in(
        f'{calc_dir}/pw.in',
        prim,
        input_data=input_scf,
        pseudopotentials=pseudos,
        kpts=kpts_scf,
    )

    # --- Write bands.in (NSCF bands) ---
    input_bands = dict(COMMON_INPUT)
    input_bands['calculation'] = 'bands'
    kpts_bands = kp_kpts(kp)
    write_espresso_in(
        f'{calc_dir}/bands.in',
        prim,
        input_data=input_bands,
        pseudopotentials=pseudos,
        kpts=kpts_bands,
    )

    # --- Write bands_post.in (bands.x input) ---
    # bands.x reads from the outdir written by pw.x
    bands_post_content = f"""&BANDS
    prefix = 'cspbbr3'
    outdir = './tmp'
    filband = '{nnn:03d}.bands.dat'
    lsym = .false.
/
"""
    with open(f'{calc_dir}/bands_post.in', 'w') as f:
        f.write(bands_post_content)

    print(f'Set up {calc_dir}: nat={nat}, ntyp={ntyp}, kpts_scf={kpts_scf}, n_kpts_bands={len(kpts_bands)}')


def main():
    for nnn in range(51):
        setup_structure(nnn)
    print('Done: all 51 calculation directories set up.')


if __name__ == '__main__':
    main()
