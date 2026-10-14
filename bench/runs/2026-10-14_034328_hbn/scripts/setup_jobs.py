#!/usr/bin/env python3
"""
setup_jobs.py -- Generate QE input files for all 51 hBN structures.

For each structure NNN (000..050):
  calculations/NNN/
    B.upf, N.upf       (pseudopotentials, copied from template/)
    kpath.json          (from chem_llm.kpath)
    pw.in               (SCF input, written by write_espresso_in)
    bands.in            (NSCF bands input, written by write_espresso_in)
    bands_post.in       (bands.x input, written manually)

Key decisions (see notes):
  - Pseudopotentials: B (z_valence=3, sr, PBESOL), N (z_valence=5, sr, PBESOL)
  - Total electrons = 2*3 + 2*5 = 16, occupied bands = 8 (nspin=1)
  - ecutwfc = 2 * 42 Ha = 84 Ry  (INVARIANT 5: 1 Ha = 2 Ry)
  - ecutrho = 4 * 84 = 336 Ry
  - occupations='fixed' (insulator, gap ~4 eV)
  - nspin=1 (non-magnetic)
  - conv_thr=1e-8
  - nbnd=20 (well above 8 occupied bands)
  - SCF k-mesh: scf_kmesh(prim, metal=False) from chem_llm.kpath
  - Bands k-path: kp_kpts(kp) from chem_llm.kpath (INVARIANT 3)
  - bands_post.in: lsym=.false. (skip band-representation file)
"""

import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# --- Pseudopotential cutoffs (INVARIANT 5) ---
# B hint: 38 Ha, N hint: 42 Ha. Use max = 42 Ha.
# 1 Ha = 2 Ry, so ecutwfc = 2 * 42 = 84 Ry.
ECUTWFC_HA = 42.0
ECUTWFC_RY = 2.0 * ECUTWFC_HA  # 84 Ry
ECUTRHO_RY = 4.0 * ECUTWFC_RY  # 336 Ry

# --- Common QE settings ---
PREFIX = 'hbn'
OUTDIR = './tmp'
CONV_THR = 1e-8
NBND = 20  # well above 8 occupied bands

# --- Pseudopotential files (relative to calculations/NNN/) ---
PSEUDO_FILES = {'B': 'B.upf', 'N': 'N.upf'}

# --- Template directory ---
TEMPLATE_DIR = 'template'

# --- Structure directory ---
STRUCT_DIR = 'structures'

# --- Calculation directory ---
CALC_DIR = 'calculations'


def write_bands_post_in(path, nnn):
    """Write the bands.x post-processing input file."""
    content = f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{nnn}.bands.dat'
    lsym = .false.
/
"""
    with open(path, 'w') as f:
        f.write(content)


def main():
    os.makedirs(CALC_DIR, exist_ok=True)

    for nnn in range(51):
        nnn_str = f'{nnn:03d}'
        calc_path = os.path.join(CALC_DIR, nnn_str)
        os.makedirs(calc_path, exist_ok=True)

        # --- Copy pseudopotentials ---
        for elem, fname in PSEUDO_FILES.items():
            src = os.path.join(TEMPLATE_DIR, fname)
            dst = os.path.join(calc_path, fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

        # --- Read structure ---
        cif_path = os.path.join(STRUCT_DIR, f'structure_{nnn_str}.cif')
        atoms = ase.io.read(cif_path)

        # --- Get k-path and primitive cell (INVARIANT 3) ---
        kp = standard_kpath(atoms)
        prim = kp_atoms(kp)

        # INVARIANT 1: nat and ntyp from the actual Atoms object
        nat = len(prim)
        ntyp = len(set(prim.get_chemical_symbols()))
        assert nat == 4, f"Structure {nnn_str}: expected 4 atoms, got {nat}"
        assert ntyp == 2, f"Structure {nnn_str}: expected 2 species, got {ntyp}"

        # --- Save kpath.json ---
        kp_save(kp, os.path.join(calc_path, 'kpath.json'))

        # --- SCF k-mesh (from chem_llm.kpath, not hand-typed) ---
        kpts_scf = scf_kmesh(prim, metal=False)

        # --- SCF input (pw.in) ---
        scf_input_data = {
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
        write_espresso_in(
            os.path.join(calc_path, 'pw.in'),
            prim,
            input_data=scf_input_data,
            pseudopotentials=PSEUDO_FILES,
            kpts=kpts_scf,
        )

        # --- Bands input (bands.in) ---
        bands_input_data = {
            'calculation': 'bands',
            'prefix': PREFIX,
            'outdir': OUTDIR,
            'pseudo_dir': './',
            'ecutwfc': ECUTWFC_RY,
            'ecutrho': ECUTRHO_RY,
            'occupations': 'fixed',
            'nbnd': NBND,
            'nspin': 1,
        }
        write_espresso_in(
            os.path.join(calc_path, 'bands.in'),
            prim,
            input_data=bands_input_data,
            pseudopotentials=PSEUDO_FILES,
            kpts=kp_kpts(kp),
        )

        # --- Bands post-processing input (bands_post.in) ---
        write_bands_post_in(os.path.join(calc_path, 'bands_post.in'), nnn_str)

        print(f"Structure {nnn_str}: done (nat={nat}, ntyp={ntyp}, kpts_scf={kpts_scf})")

    print(f"\nAll 51 calculation directories created under {CALC_DIR}/")


if __name__ == '__main__':
    main()
