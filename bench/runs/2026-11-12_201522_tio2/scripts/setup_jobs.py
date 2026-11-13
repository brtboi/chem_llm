"""
setup_jobs.py -- Generate QE input files for all 51 TiO2 structures.

For each structure NNN (000-050):
  calculations/NNN/
    Ti.pbesol_upf, O.pbesol_upf   (copied from template/)
    kpath.json                    (from chem_llm.dft.kpath)
    pw.in                         (SCF input, written by write_espresso_in)
    bands.in                      (NSCF bands input, written by write_espresso_in)
    bands_post.in                 (bands.x input, written by write_fortran_namelist)

Key decisions (see notes):
  - ecutwfc = 92 Ry  (max of Ti 42 Ha, O 46 Ha, converted: 46*2 = 92 Ry)
  - ecutrho = 368 Ry (4 * ecutwfc)
  - occupations = 'fixed' (semiconductor, gap ~1.77 eV)
  - conv_thr = 1e-8 Ry
  - nspin = 1 (non-magnetic)
  - nbnd = 40 (occupied bands = 48/2 = 24, padded)
  - SCF k-mesh from scf_kmesh(prim, metal=False)
  - Bands k-points from kp_kpts(kp) (304 points along standard path)
  - lsym = .false. in bands_post.in (skip band-representation file)
"""

import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in, write_fortran_namelist
from chem_llm.dft.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# --- Constants ---
N_STRUCTURES = 51
TEMPLATE_DIR = 'template'
STRUCTURES_DIR = 'structures'
CALC_DIR = 'calculations'

# Pseudopotential files
PSEUDO_FILES = ['Ti.pbesol_upf', 'O.pbesol_upf']

# Cutoffs (Ry) -- INVARIANT 5: converted from Ha (1 Ha = 2 Ry)
# Ti: 42 Ha -> 84 Ry, O: 46 Ha -> 92 Ry. Use max = 92 Ry.
ECUTWFC = 92.0   # Ry
ECUTRHO = 368.0  # Ry (4 * ecutwfc)

# Other settings
PREFIX = 'tio2'
CONV_THR = 1e-8  # Ry
NBND = 40        # bands (occupied = 24, padded)

# Pseudopotential mapping for write_espresso_in
PSEUDO_MAP = {
    'Ti': 'Ti.pbesol_upf',
    'O': 'O.pbesol_upf',
}


def make_calc_dir(nnn: str) -> str:
    """Create calculations/NNN/ and copy pseudopotentials."""
    d = os.path.join(CALC_DIR, nnn)
    os.makedirs(d, exist_ok=True)
    for pf in PSEUDO_FILES:
        src = os.path.join(TEMPLATE_DIR, pf)
        dst = os.path.join(d, pf)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    return d


def write_pw_in(path: str, atoms, kpts, outdir: str):
    """Write SCF pw.x input file using write_espresso_in."""
    input_data = {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': outdir,
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC,
        'ecutrho': ECUTRHO,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
        'nspin': 1,
    }
    with open(path, 'w') as f:
        write_espresso_in(f, atoms, input_data=input_data,
                          pseudopotentials=PSEUDO_MAP, kpts=kpts)


def write_bands_in(path: str, atoms, kpts, outdir: str):
    """Write NSCF bands pw.x input file using write_espresso_in."""
    input_data = {
        'calculation': 'bands',
        'prefix': PREFIX,
        'outdir': outdir,
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC,
        'ecutrho': ECUTRHO,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
        'nspin': 1,
        'nbnd': NBND,
    }
    with open(path, 'w') as f:
        write_espresso_in(f, atoms, input_data=input_data,
                          pseudopotentials=PSEUDO_MAP, kpts=kpts)


def write_bands_post_in(path: str, nnn: str, outdir: str):
    """Write bands.x post-processing input file."""
    input_data = {
        'prefix': PREFIX,
        'outdir': outdir,
        'filband': f'{nnn}.bands.dat',
        'lsym': False,
    }
    with open(path, 'w') as f:
        write_fortran_namelist(f, input_data=input_data, binary='bands')


def main():
    for i in range(N_STRUCTURES):
        nnn = f'{i:03d}'
        cif_path = os.path.join(STRUCTURES_DIR, f'structure_{nnn}.cif')
        atoms = ase.io.read(cif_path)

        # INVARIANT 1: nat/ntyp from the actual Atoms object
        nat = len(atoms)
        ntyp = len(set(atoms.get_chemical_symbols()))
        assert nat == 6, f'{nnn}: expected 6 atoms, got {nat}'
        assert ntyp == 2, f'{nnn}: expected 2 species, got {ntyp}'

        # INVARIANT 3: k-path from chem_llm.dft.kpath for THIS structure
        kp = standard_kpath(atoms)
        prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in

        # SCF k-mesh (Monkhorst-Pack, non-metal)
        scf_kpts = scf_kmesh(prim, metal=False)

        # Bands k-points from the standard path
        bands_kpts = kp_kpts(kp)  # shape (n_kpts, 4)

        # Create directory and copy pseudopotentials
        calc_dir = make_calc_dir(nnn)
        outdir = os.path.join('tmp')  # relative to calculations/NNN/

        # Save kpath.json
        kp_save(kp, os.path.join(calc_dir, 'kpath.json'))

        # Write input files
        pw_in_path = os.path.join(calc_dir, 'pw.in')
        bands_in_path = os.path.join(calc_dir, 'bands.in')
        bands_post_path = os.path.join(calc_dir, 'bands_post.in')

        write_pw_in(pw_in_path, prim, scf_kpts, outdir)
        write_bands_in(bands_in_path, prim, bands_kpts, outdir)
        write_bands_post_in(bands_post_path, nnn, outdir)

        print(f'{nnn}: done (nat={nat}, ntyp={ntyp}, kpath={kp["path"]})')

    print(f'\nAll {N_STRUCTURES} calculation directories created.')


if __name__ == '__main__':
    main()
