"""
setup_jobs.py — Create QE input files for all 51 GaAs structures.

For each structure NNN (000-050):
  calculations/NNN/
    Ga.pbesol-srn-rrkjus_psl.1.0.0.UPF   (copied from template/)
    As.pbesol-srn-rrkjus_psl.1.0.0.UPF   (copied from template/)
    kpath.json                            (from chem_llm.kpath)
    pw.in                                 (SCF input, written by write_espresso_in)
    bands.in                              (bands input, written by write_espresso_in)
    bands_post.in                         (bands.x input, written manually)

Key decisions (see notes):
  - ecutwfc = 114 Ry (2 * 57 Ha, max of Ga=57 Ha and As=52 Ha; INVARIANT 5)
  - ecutrho = 456 Ry (4 * ecutwfc)
  - occupations = 'fixed' (GaAs is a semiconductor, ~1.4 eV gap)
  - conv_thr = 1e-8 Ry
  - SCF k-mesh: 12x12x12 Monkhorst-Pack
  - nbnd = 48 (occupied bands = 22 from z_valence sum 44/2; padded to 48)
  - Cell for pw.in and bands.in: kp_atoms(kp) (INVARIANT 3)
  - kpts for bands.in: kp_kpts(kp) (INVARIANT 3)
"""

import os
import shutil
import numpy as np
import ase.io
from ase.io import read, write
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save

# Pseudopotential files (relative to template/)
PP_GA = 'Ga.pbesol-srn-rrkjus_psl.1.0.0.UPF'
PP_AS = 'As.pbesol-srn-rrkjus_psl.1.0.0.UPF'

# Cutoffs in Ry (converted from Ha per INVARIANT 5: 1 Ha = 2 Ry)
# Ga: 57 Ha -> 114 Ry; As: 52 Ha -> 104 Ry; use max = 114 Ry
ECUTWFC_RY = 114.0
ECUTRHO_RY = 4 * ECUTWFC_RY  # 456 Ry

# SCF settings
CONV_THR = 1e-8  # Ry
K_MESH = (12, 12, 12)  # Monkhorst-Pack for SCF

# Bands settings
NBND = 48  # occupied bands = 22 (44 electrons / 2); padded to 48

# Pseudopotential valence electrons (from get_pseudopotential results)
# Ga: z_valence = 21, As: z_valence = 23
# Total = 44 electrons, occupied bands = 22
Z_VAL_GA = 21
Z_VAL_AS = 23
TOTAL_ELECTRONS = Z_VAL_GA + Z_VAL_AS  # 44
OCCUPIED_BANDS = TOTAL_ELECTRONS // 2  # 22
assert NBND > OCCUPIED_BANDS, f'nbnd={NBND} must be > occupied bands={OCCUPIED_BANDS}'

# Pseudopotential filenames for write_espresso_in
PSEUDO_FILES = {
    'Ga': PP_GA,
    'As': PP_AS,
}

# Prefix and outdir (relative to calculations/NNN/)
PREFIX = 'gaas'
OUTDIR = './tmp'

# pseudo_dir: pw.x runs inside calculations/NNN/, pseudos are in template/
# We copy the PP files into calculations/NNN/ so pseudo_dir='.' works
PSEUDO_DIR = '.'


def write_bands_post_in(path, prefix, outdir, filband):
    """Write a bands.x input file (bands_post.in).
    
    bands.x uses a simple namelist format:
      &BANDS
        prefix = '...'
        outdir = '...'
        filband = '...'
      /
    """
    with open(path, 'w') as f:
        f.write('&BANDS\n')
        f.write(f"  prefix = '{prefix}'\n")
        f.write(f"  outdir = '{outdir}'\n")
        f.write(f"  filband = '{filband}'\n")
        f.write('/\n')
        f.write('EOF\n')


def setup_one_structure(nnn, base_atoms):
    """Set up calculations/NNN/ for one structure."""
    calc_dir = f'calculations/{nnn}'
    os.makedirs(calc_dir, exist_ok=True)

    # Copy pseudopotential files into the calculation directory
    for pp_file in [PP_GA, PP_AS]:
        src = os.path.join('template', pp_file)
        dst = os.path.join(calc_dir, pp_file)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Read the structure
    atoms = read(f'structures/structure_{nnn}.cif')

    # INVARIANT 1: nat and ntyp from the actual ASE Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f'Structure {nnn}: expected nat=2, got {nat}'
    assert ntyp == 2, f'Structure {nnn}: expected ntyp=2, got {ntyp}'

    # INVARIANT 3: get the standard k-path for THIS structure
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    kpts_bands = kp_kpts(kp)  # k-points for bands.in

    # Save kpath.json (holds labels, ticks, segments)
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))

    # --- SCF input (pw.in) ---
    # Flat input_data dict; ASE routes each key to the correct namelist (INVARIANT 6)
    scf_input_data = {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': PSEUDO_DIR,
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
    }
    # kpts as a tuple -> Monkhorst-Pack grid
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=scf_input_data,
        pseudopotentials=PSEUDO_FILES,
        kpts=K_MESH,
    )

    # --- Bands input (bands.in) ---
    bands_input_data = {
        'calculation': 'bands',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': PSEUDO_DIR,
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'nbnd': NBND,
    }
    # kpts from kp_kpts(kp) — a list of (kx, ky, kz, weight) tuples
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=bands_input_data,
        pseudopotentials=PSEUDO_FILES,
        kpts=kpts_bands,
    )

    # --- bands.x post-processing input (bands_post.in) ---
    filband = f'{nnn}.bands.dat'
    write_bands_post_in(
        os.path.join(calc_dir, 'bands_post.in'),
        PREFIX, OUTDIR, filband,
    )

    print(f'Set up {calc_dir}/ (nat={nat}, ntyp={ntyp}, k-path: {kp.path if hasattr(kp, "path") else "ok"})')


def main():
    # Read the base structure to get the formula (for sanity check)
    base = read('structures/structure_000.cif')
    print(f'Base structure: {len(base)} atoms, symbols = {base.get_chemical_symbols()}')

    # Set up all 51 structures
    for i in range(51):
        nnn = f'{i:03d}'
        setup_one_structure(nnn, base)

    print('Done. Set up calculations/000/ through calculations/050/')


if __name__ == '__main__':
    main()
