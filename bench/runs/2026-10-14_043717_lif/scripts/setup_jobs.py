#!/usr/bin/env python3
"""
setup_jobs.py -- Generate QE input files for all 51 LiF structures.

For each structure NNN (000-050):
  calculations/NNN/
    Li.upf, F.upf       -- pseudopotentials (copied from template/)
    kpath.json          -- standard k-path from chem_llm.kpath
    pw.in               -- SCF input (write_espresso_in)
    bands.in            -- bands input (write_espresso_in, kpts from kpath)
    bands_post.in       -- bands.x post-processing input

Key decisions (see notes):
  - ecutwfc = 96 Ry (from Li's 48 Ha recommendation, 1 Ha = 2 Ry)
  - ecutrho = 4 * ecutwfc = 384 Ry
  - occupations = 'fixed' (LiF is an insulator, gap ~8.7 eV)
  - conv_thr = 1e-8 (in &ELECTRONS, routed by ASE)
  - nspin = 1 (non-magnetic)
  - nbnd = 12 (total electrons = 10, occupied = 5, pad above)
  - SCF k-mesh from scf_kmesh(prim, metal=False)
  - bands k-points from kp_kpts(kp) (INVARIANT 3)
  - lsym = .false. in bands_post.in (skip band-representation file)
"""

import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# Pseudopotential files (relative to calculations/NNN)
PP_Li = 'Li.upf'
PP_F = 'F.upf'

# Cutoffs (Ry) -- INVARIANT 5: 1 Ha = 2 Ry
# Li: 48 Ha -> 96 Ry; F: 42 Ha -> 84 Ry; use the larger
ECUTWFC_RY = 96.0
ECUTRHO_RY = 4.0 * ECUTWFC_RY  # 384 Ry

# Electron count from pseudopotential z_valence (INVARIANT 2)
# Li: z_valence=3, F: z_valence=7 -> total = 10 electrons
# Spin-unpolarized: occupied bands = 10/2 = 5
# nbnd must be >= occupied bands; pad to 12
NBND = 12

# Common input_data for SCF and bands (flat dict; ASE routes to correct namelists)
# INVARIANT 6: do NOT hand-sort into &CONTROL / &SYSTEM / &ELECTRONS
COMMON_INPUT = {
    'prefix': 'lif',
    'outdir': 'tmp',
    'pseudo_dir': '.',
    'ecutwfc': ECUTWFC_RY,
    'ecutrho': ECUTRHO_RY,
    'occupations': 'fixed',
    'conv_thr': 1e-8,
    'nspin': 1,
}

# Pseudopotential mapping for write_espresso_in
PSEUDOS = {'Li': PP_Li, 'F': PP_F}


def setup_one(nnn: str, base_atoms):
    """Set up calculations/NNN/ for one structure."""
    calc_dir = f'calculations/{nnn}'
    os.makedirs(calc_dir, exist_ok=True)

    # Copy pseudopotentials into calc dir (pseudo_dir='.')
    for pp in (PP_Li, PP_F):
        src = f'template/{pp}'
        dst = f'{calc_dir}/{pp}'
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Read the structure
    cif_path = f'structures/structure_{nnn}.cif'
    atoms = ase.io.read(cif_path)

    # INVARIANT 1: nat/ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"{nnn}: expected 2 atoms, got {nat}"
    assert ntyp == 2, f"{nnn}: expected 2 species, got {ntyp}"

    # INVARIANT 3: k-path and cell from chem_llm.kpath for THIS structure
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    kp_save(kp, f'{calc_dir}/kpath.json')

    # SCF k-mesh (RULE: never typed by hand)
    kpts_scf = scf_kmesh(prim, metal=False)

    # --- pw.in (SCF) ---
    scf_input = dict(COMMON_INPUT)
    scf_input['calculation'] = 'scf'
    with open(f'{calc_dir}/pw.in', 'w') as f:
        write_espresso_in(f, prim, input_data=scf_input,
                          pseudopotentials=PSEUDOS, kpts=kpts_scf)

    # --- bands.in (NSCF bands) ---
    bands_input = dict(COMMON_INPUT)
    bands_input['calculation'] = 'bands'
    bands_input['nbnd'] = NBND
    kpts_bands = kp_kpts(kp)  # ndarray of shape (nkpts, 3)
    with open(f'{calc_dir}/bands.in', 'w') as f:
        write_espresso_in(f, prim, input_data=bands_input,
                          pseudopotentials=PSEUDOS, kpts=kpts_bands)

    # --- bands_post.in (bands.x) ---
    # bands.x input: &BANDS namelist with prefix, outdir, filband, lsym
    # Write manually (write_espresso_in is for pw.x; bands.x uses write_fortran_namelist)
    from ase.io.espresso import write_fortran_namelist
    bands_post_data = {
        'BANDS': {
            'prefix': 'lif',
            'outdir': 'tmp',
            'filband': f'{nnn}.bands.dat',
            'lsym': False,
        }
    }
    with open(f'{calc_dir}/bands_post.in', 'w') as f:
        write_fortran_namelist(f, input_data=bands_post_data, binary='bands')

    print(f"{nnn}: nat={nat}, ntyp={ntyp}, kpath={kp['path']}, "
          f"scf_kpts={kpts_scf}, nbnd={NBND}")


def main():
    # Read base structure to verify
    base = ase.io.read('structures/structure_000.cif')
    print(f"Base structure: {len(base)} atoms, {base.get_chemical_symbols()}")

    for i in range(51):
        nnn = f'{i:03d}'
        setup_one(nnn, base)

    print("\nDone. All 51 calculation directories set up.")


if __name__ == '__main__':
    main()
