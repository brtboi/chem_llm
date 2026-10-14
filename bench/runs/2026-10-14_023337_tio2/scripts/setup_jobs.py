#!/usr/bin/env python3
"""
setup_jobs.py -- Generate QE input files for all 51 TiO2 structures.

For each structure NNN (000-050), creates calculations/NNN/ containing:
  - Ti.upf, O.upf (copied from template/)
  - kpath.json (from chem_llm.kpath)
  - pw.in (SCF calculation)
  - bands.in (NSCF band-structure calculation)
  - bands_post.in (bands.x post-processing input)

Key decisions (see notes):
  - ecutwfc = 92.0 Ry (from max(42.0, 46.0) Ha * 2 Ry/Ha)
  - ecutrho = 368.0 Ry (4 * ecutwfc)
  - occupations = 'fixed' (semiconductor, gap ~1.77 eV)
  - conv_thr = 1e-8 Ry
  - nbnd = 40 (pads above 24 occupied bands for 48 electrons)
  - SCF k-mesh: computed via scf_kmesh(prim, metal=False)
  - Bands k-path: from chem_llm.kpath (INVARIANT 3)
  - lsym = .false. in bands_post.in (skip band-representation file)
"""

import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in

from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# --- Pseudopotential files ---
PP_DIR = 'template'
PP_FILES = {'Ti': 'Ti.upf', 'O': 'O.upf'}

# --- Cutoffs (converted from Ha to Ry per INVARIANT 5) ---
# Ti: 42.0 Ha, O: 46.0 Ha -> max = 46.0 Ha = 92.0 Ry
ECUTWFC_RY = 92.0
ECUTRHO_RY = 4 * ECUTWFC_RY  # 368.0 Ry

# --- Other settings ---
PREFIX = 'tio2'
OUTDIR = 'tmp'
CONV_THR = 1e-8  # Ry
NBND = 40  # pads above 24 occupied bands (48 electrons / 2)

# --- Pseudopotential dict for write_espresso_in ---
# pseudo_dir='.' means look in the current directory (calculations/NNN/)
pseudos = {sym: fname for sym, fname in PP_FILES.items()}


def write_bands_post_in(nnn: str, outdir: str = OUTDIR, prefix: str = PREFIX):
    """Write bands_post.in (bands.x input) for structure NNN.
    
    bands.x reads the NSCF wavefunctions from outdir and writes
    the band energies to filband. lsym=.false. skips the band-
    representation file (not needed for plotting).
    """
    filband = f'{nnn}.bands.dat'
    content = f"""&BANDS
    prefix = '{prefix}'
    outdir = '{outdir}'
    filband = '{filband}'
    lsym = .false.
/
"""
    with open(f'calculations/{nnn}/bands_post.in', 'w') as f:
        f.write(content)


def main():
    for nnn in range(51):
        nnn_str = f'{nnn:03d}'
        calc_dir = f'calculations/{nnn_str}'
        os.makedirs(calc_dir, exist_ok=True)

        # Copy pseudopotentials into the calculation directory
        for sym, fname in PP_FILES.items():
            src = os.path.join(PP_DIR, fname)
            dst = os.path.join(calc_dir, fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

        # Read the structure
        atoms = ase.io.read(f'structures/structure_{nnn_str}.cif')

        # Get the standard k-path and primitive cell (INVARIANT 3)
        kp = standard_kpath(atoms)
        prim = kp_atoms(kp)

        # Save kpath.json for plotting later
        kp_save(kp, os.path.join(calc_dir, 'kpath.json'))

        # --- SCF k-mesh (Monkhorst-Pack, computed not hand-typed) ---
        kpts_scf = scf_kmesh(prim, metal=False)

        # --- SCF input (pw.in) ---
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
            os.path.join(calc_dir, 'pw.in'),
            prim,
            input_data=input_data_scf,
            pseudopotentials=pseudos,
            kpts=kpts_scf,
        )

        # --- Bands input (bands.in) ---
        # kpts = the k-path from chem_llm.kpath (INVARIANT 3)
        kpts_bands = kp_kpts(kp)
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
            os.path.join(calc_dir, 'bands.in'),
            prim,
            input_data=input_data_bands,
            pseudopotentials=pseudos,
            kpts=kpts_bands,
        )

        # --- Bands post-processing input (bands_post.in) ---
        write_bands_post_in(nnn_str)

        print(f'Wrote {calc_dir}/ (pw.in, bands.in, bands_post.in, kpath.json, pseudos)')

    print('\nDone: set up all 51 calculation directories.')


if __name__ == '__main__':
    main()
