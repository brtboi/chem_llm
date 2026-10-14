#!/usr/bin/env python3
"""
setup_jobs.py - Generate QE input files for all 51 NaCl structures.

For each structure NNN (000-050), creates calculations/NNN/ containing:
- Pseudopotential files (copied from template/)
- kpath.json (from chem_llm.kpath)
- pw.in (SCF input)
- bands.in (NSCF bands input)
- bands_post.in (bands.x post-processing input)

Key settings (from STEP 4 note):
- ecutwfc = 88.0 Ry (max of Na 44 Ha * 2, Cl 29 Ha * 2)
- ecutrho = 352.0 Ry (4 * ecutwfc)
- occupations = 'fixed' (insulator, gap ~5 eV)
- conv_thr = 1.0e-8 Ry
- nspin = 1 (non-magnetic)
- nbnd = 16 (2x occupied bands = 8)
- prefix = 'nacl', outdir = './tmp', pseudo_dir = './'
"""

import os
import shutil
import numpy as np
from ase.io import read
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# --- Pseudopotential files ---
PP_FILES = {
    'Na': 'Na.pbesol-rrkjus_psl.1.0.0.UPF',
    'Cl': 'Cl.pbesol-rrkjus_psl.1.0.0.UPF',
}

# --- QE settings ---
ECUTWFC_RY = 88.0  # Ry (max of Na 44 Ha * 2, Cl 29 Ha * 2)
ECUTRHO_RY = 352.0  # Ry (4 * ecutwfc)
CONV_THR = 1.0e-8  # Ry
NBND = 16  # 2x occupied bands (8 occupied for 16 valence electrons)
PREFIX = 'nacl'
OUTDIR = './tmp'

# --- INVARIANT 5: unit conversion check ---
# Na: 44 Ha * 2 = 88 Ry
# Cl: 29 Ha * 2 = 58 Ry
# max = 88 Ry
assert ECUTWFC_RY == 88.0, f"ecutwfc should be 88.0 Ry, got {ECUTWFC_RY}"
assert ECUTRHO_RY == 4 * ECUTWFC_RY, f"ecutrho should be 4*ecutwfc, got {ECUTRHO_RY}"


def setup_structure(nnn: str, atoms: 'ase.Atoms'):
    """Set up QE input files for one structure."""
    calc_dir = f'calculations/{nnn}'
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(f'{calc_dir}/tmp', exist_ok=True)

    # Copy pseudopotential files into calc_dir
    for sym, pp_file in PP_FILES.items():
        src = f'template/{pp_file}'
        dst = f'{calc_dir}/{pp_file}'
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # --- INVARIANT 3: k-path from chem_llm.kpath ---
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    kp_save(kp, f'{calc_dir}/kpath.json')

    # --- INVARIANT 1: nat/ntyp from actual Atoms object ---
    nat = len(prim)
    ntyp = len(set(prim.get_chemical_symbols()))
    assert nat > 0, f"Structure {nnn}: nat must be > 0"
    assert ntyp > 0, f"Structure {nnn}: ntyp must be > 0"

    # --- SCF k-point mesh (INVARIANT: never typed by hand) ---
    kpts_scf = scf_kmesh(prim, metal=False)

    # --- Pseudopotential dict for write_espresso_in ---
    pseudopotentials = {sym: PP_FILES[sym] for sym in prim.get_chemical_symbols()}

    # --- SCF input (pw.in) ---
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
    write_espresso_in(
        f'{calc_dir}/pw.in',
        prim,
        input_data=input_data_scf,
        pseudopotentials=pseudopotentials,
        kpts=kpts_scf,
    )

    # --- Bands input (bands.in) ---
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
    write_espresso_in(
        f'{calc_dir}/bands.in',
        prim,
        input_data=input_data_bands,
        pseudopotentials=pseudopotentials,
        kpts=kp_kpts(kp),
    )

    # --- Bands post-processing input (bands_post.in) ---
    # bands.x input: &BANDS namelist
    bands_post_content = f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{nnn}.bands.dat'
    lsym = .false.
/
"""
    with open(f'{calc_dir}/bands_post.in', 'w') as f:
        f.write(bands_post_content)

    print(f"Setup complete for {calc_dir}")


def main():
    """Set up all 51 structures."""
    for i in range(51):
        nnn = f'{i:03d}'
        cif_path = f'structures/structure_{nnn}.cif'
        atoms = read(cif_path)
        setup_structure(nnn, atoms)

    print("\nAll 51 structures set up.")


if __name__ == '__main__':
    main()
