"""
setup_jobs.py -- Create calculations/NNN/ directories for all 51 structures,
each containing:
  - C.upf (pseudopotential, copied from template/)
  - kpath.json (from chem_llm.kpath)
  - pw.in (SCF input, written with write_espresso_in)
  - bands.in (NSCF bands input, written with write_espresso_in)
  - bands_post.in (bands.x post-processing input)

All QE input variables are decided per the STEP 4 note:
  - ecutwfc = 82.0 Ry  (2.0 * 41.0 Ha, INVARIANT 5)
  - ecutrho = 328.0 Ry (4 * ecutwfc)
  - occupations = 'smearing', smearing = 'mv', degauss = 0.01 Ry
    (graphite is a semimetal, band_gap_eV = 0.0)
  - conv_thr = 1e-8
  - SCF k-mesh: scf_kmesh(prim, metal=True) from chem_llm.kpath
  - nbnd = 20 for bands.in (16 electrons -> 8 occupied bands, padded)
  - prefix = 'graphite', outdir = 'tmp', pseudo_dir = '.'
  - bands_post.in: &BANDS prefix='graphite', outdir='tmp', filband='NNN.bands.dat', lsym=.false.
"""

import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# --- Constants ---
N_STRUCTURES = 51
PREFIX = 'graphite'
OUTDIR = 'tmp'
ECUTWFC_RY = 82.0   # Ry (2.0 * 41.0 Ha per INVARIANT 5)
ECUTRHO_RY = 328.0  # Ry (4 * ecutwfc)
CONV_THR = 1e-8
NBND = 20           # 16 electrons -> 8 occupied bands, padded well above
DEGAUSS = 0.01      # Ry, smearing width for semimetal

# Pseudopotential file (relative to calculations/NNN/)
PSEUDO_FILE = 'C.upf'

# --- Copy pseudopotential into template for reference ---
# (already fetched to template/C.upf in STEP 3)

for nnn in range(N_STRUCTURES):
    calc_dir = f'calculations/{nnn:03d}'
    os.makedirs(calc_dir, exist_ok=True)

    # Copy pseudopotential into the calculation directory
    # (pseudo_dir='.' so pw.x looks in the current directory)
    shutil.copy('template/C.upf', os.path.join(calc_dir, PSEUDO_FILE))

    # Read the structure
    atoms = ase.io.read(f'structures/structure_{nnn:03d}.cif')

    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 4, f'structure_{nnn:03d}: nat={nat}, expected 4'
    assert ntyp == 1, f'structure_{nnn:03d}: ntyp={ntyp}, expected 1'

    # INVARIANT 3: k-path from chem_llm.kpath for THIS structure
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))

    # SCF k-mesh (Monkhorst-Pack, metal=True for semimetal)
    scf_kpts = scf_kmesh(prim, metal=True)

    # --- Write pw.in (SCF) ---
    scf_input_data = {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'smearing',
        'smearing': 'mv',
        'degauss': DEGAUSS,
        'conv_thr': CONV_THR,
        'nat': nat,
        'ntyp': ntyp,
    }
    with open(os.path.join(calc_dir, 'pw.in'), 'w') as f:
        write_espresso_in(f, prim, input_data=scf_input_data,
                          pseudopotentials={'C': PSEUDO_FILE},
                          kpts=scf_kpts)

    # --- Write bands.in (NSCF bands) ---
    bands_input_data = {
        'calculation': 'bands',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'smearing',
        'smearing': 'mv',
        'degauss': DEGAUSS,
        'conv_thr': CONV_THR,
        'nat': nat,
        'ntyp': ntyp,
        'nbnd': NBND,
    }
    with open(os.path.join(calc_dir, 'bands.in'), 'w') as f:
        write_espresso_in(f, prim, input_data=bands_input_data,
                          pseudopotentials={'C': PSEUDO_FILE},
                          kpts=kp_kpts(kp))

    # --- Write bands_post.in (bands.x input) ---
    # bands.x uses a simple fortran namelist &BANDS
    bands_post_content = f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{nnn:03d}.bands.dat'
    lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)

    print(f'calculations/{nnn:03d}: done (nat={nat}, ntyp={ntyp}, scf_kpts={scf_kpts})')

print(f'\nAll {N_STRUCTURES} calculation directories created.')
print('Done.')
