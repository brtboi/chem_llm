#!/usr/bin/env python3
"""
setup_jobs.py -- Generate QE input files for all 51 LiF structures.

For each structure NNN (000-050), creates calculations/NNN/ containing:
  - Pseudopotential files (copied from template/)
  - kpath.json (from chem_llm.kpath)
  - pw.in (SCF input)
  - bands.in (NSCF bands input)
  - bands_post.in (bands.x post-processing input)

Key settings (from STEP 4 note):
  - ecutwfc = 96.0 Ry (48 Ha * 2, INVARIANT 5)
  - ecutrho = 384.0 Ry (4 * ecutwfc)
  - occupations = 'fixed' (LiF is an insulator)
  - conv_thr = 1e-8
  - nbnd = 16 (total valence e- = 10, occupied = 5, safety margin)
  - nspin = 1, no magnetism, no spin-orbit (Z < 54)
  - prefix = 'lif', outdir = './tmp', pseudo_dir = '.'
"""

import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# --- Constants ---
N_STRUCTURES = 51
PREFIX = 'lif'
OUTDIR = './tmp'

# Pseudopotential files (relative to calculations/NNN/)
PP_Li = 'Li.pbesol_s-rrkjus_psl.1.0.0.UPF'
PP_F = 'F.pbesol_s-rrkjus_psl.1.0.0.UPF'

# Cutoffs (INVARIANT 5: convert from Ha to Ry, 1 Ha = 2 Ry)
# Li: 48 Ha, F: 42 Ha -> use max = 48 Ha = 96 Ry
ECUTWFC_RY = 96.0
ECUTRHO_RY = 4.0 * ECUTWFC_RY  # 384 Ry

# nbnd: total valence electrons = 3(Li) + 7(F) = 10, occupied bands = 5
# Pad to 16 for safety
NBND = 16

# conv_thr in &ELECTRONS (not &CONTROL -- INVARIANT 6)
CONV_THR = 1e-8


def setup_one(nnn: str, atoms: ase.Atoms):
    """Set up calculations/NNN/ for one structure."""
    calc_dir = f'calculations/{nnn}'
    os.makedirs(calc_dir, exist_ok=True)

    # Copy pseudopotentials into the calculation directory
    for pp in [PP_Li, PP_F]:
        src = f'template/{pp}'
        dst = f'{calc_dir}/{pp}'
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Get the standard k-path for THIS structure
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in (INVARIANT 3)
    kp_save(kp, f'{calc_dir}/kpath.json')

    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(prim)
    ntyp = len(set(prim.get_chemical_symbols()))
    assert nat == 2, f"Structure {nnn}: expected nat=2, got {nat}"
    assert ntyp == 2, f"Structure {nnn}: expected ntyp=2, got {ntyp}"

    # Pseudopotential mapping
    pseudos = {'Li': PP_Li, 'F': PP_F}

    # --- SCF input (pw.in) ---
    # SCF k-mesh: computed from the primitive cell (never typed by hand)
    kpts_scf = scf_kmesh(prim, metal=False)

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

    pw_in_path = f'{calc_dir}/pw.in'
    with open(pw_in_path, 'w') as f:
        write_espresso_in(f, prim, input_data=input_data_scf,
                          pseudopotentials=pseudos, kpts=kpts_scf)

    # --- NSCF bands input (bands.in) ---
    # kpts from the k-path (INVARIANT 3)
    kpts_bands = kp_kpts(kp)

    input_data_bands = {
        'calculation': 'nscf',
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

    bands_in_path = f'{calc_dir}/bands.in'
    with open(bands_in_path, 'w') as f:
        write_espresso_in(f, prim, input_data=input_data_bands,
                          pseudopotentials=pseudos, kpts=kpts_bands)

    # --- bands.x post-processing input (bands_post.in) ---
    # Written as a plain namelist (bands.x has no ASE writer)
    bands_post_path = f'{calc_dir}/bands_post.in'
    with open(bands_post_path, 'w') as f:
        f.write('&BANDS\n')
        f.write(f"  prefix = '{PREFIX}'\n")
        f.write(f"  outdir = '{OUTDIR}'\n")
        f.write(f"  filband = '{nnn}.bands.dat'\n")
        f.write('  lsym = .false.\n')
        f.write('/\n')

    print(f"  {nnn}: nat={nat}, ntyp={ntyp}, kpts_scf={kpts_scf}, "
          f"n_kpts_bands={len(kpts_bands)}")


def main():
    print("Setting up QE input files for all 51 structures...")
    for i in range(N_STRUCTURES):
        nnn = f'{i:03d}'
        cif_path = f'structures/structure_{nnn}.cif'
        atoms = ase.io.read(cif_path)
        setup_one(nnn, atoms)
    print("Done.")


if __name__ == '__main__':
    main()
