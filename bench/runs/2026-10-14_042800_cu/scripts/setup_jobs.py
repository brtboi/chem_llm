"""
setup_jobs.py -- Generate QE input files for all 51 Cu structures.

For each structure NNN (000-050), creates calculations/NNN/ containing:
  - Cu.pbesol-sr-rrkjus_psl.1.0.0.UPF  (pseudopotential, copied from template/)
  - kpath.json                          (k-path from chem_llm.kpath)
  - pw.in                               (SCF input, written by write_espresso_in)
  - bands.in                            (NSCF bands input, written by write_espresso_in)
  - bands_post.in                       (bands.x post-processing input)

Key settings (from STEP 4 note):
  - ecutwfc = 104.0 Ry  (52.0 Ha * 2, INVARIANT 5)
  - ecutrho = 416.0 Ry  (4 * ecutwfc)
  - occupations = 'smearing', smearing = 'mv', degauss = 0.01 Ry (Cu is a metal)
  - conv_thr = 1.0e-8 Ry
  - nbnd = 40 (19 electrons -> 10 occupied bands, well above)
  - SCF k-mesh via scf_kmesh(prim, metal=True)
  - Bands kpts via kp_kpts(kp)
  - pseudo_dir = '.' so QE looks in the current working directory (calculations/NNN)
  - bands_post.in: &BANDS prefix='cu', outdir='tmp', filband='NNN.bands.dat', lsym=.false.
"""

import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# Pseudopotential file (relative to calculations/NNN, so pseudo_dir='.')
PP_FILE = 'Cu.pbesol-sr-rrkjus_psl.1.0.0.UPF'
PP_SRC = os.path.join('template', PP_FILE)

# QE settings
ECUTWFC_RY = 104.0   # 52.0 Ha * 2 (INVARIANT 5: 1 Ha = 2 Ry)
ECUTRHO_RY = 416.0   # 4 * ecutwfc
CONV_THR = 1.0e-8    # Ry
DEGAUSS = 0.01       # Ry
NBND = 40            # 19 electrons -> 10 occupied bands, well above
PREFIX = 'cu'
OUTDIR = 'tmp'

# Verify pseudopotential exists
assert os.path.exists(PP_SRC), f"Pseudopotential not found: {PP_SRC}"

for nnn in range(51):
    label = f'{nnn:03d}'
    calc_dir = f'calculations/{label}'
    os.makedirs(calc_dir, exist_ok=True)

    # Copy pseudopotential into the calculation directory
    pp_dest = os.path.join(calc_dir, PP_FILE)
    if not os.path.exists(pp_dest):
        shutil.copy2(PP_SRC, pp_dest)

    # Read the structure
    cif_path = f'structures/structure_{label}.cif'
    atoms = ase.io.read(cif_path)

    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 1, f"Expected 1 atom, got {nat}"
    assert ntyp == 1, f"Expected 1 type, got {ntyp}"

    # Get the k-path and primitive cell (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))

    # SCF k-mesh (metal=True for Cu)
    kpts_scf = scf_kmesh(prim, metal=True)

    # Pseudopotential mapping: species -> filename (relative to calc_dir)
    pseudos = {'Cu': PP_FILE}

    # --- Write pw.in (SCF) ---
    # pseudo_dir='.' tells QE to look for pseudopotentials in the current
    # working directory (calculations/NNN), where the UPF file was copied.
    # Without this, QE falls back to $ESPRESSO_PSEUDO which does not contain
    # our Pseudo-Dojo file.
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
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=scf_input_data,
        pseudopotentials=pseudos,
        kpts=kpts_scf,
    )

    # --- Write bands.in (NSCF bands) ---
    bands_input_data = {
        'calculation': 'bands',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'nbnd': NBND,
        'nat': nat,
        'ntyp': ntyp,
    }
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=bands_input_data,
        pseudopotentials=pseudos,
        kpts=kp_kpts(kp),
    )

    # --- Write bands_post.in (bands.x input) ---
    # bands.x reads a simple namelist &BANDS with prefix, outdir, filband, lsym
    bands_post_content = f"""&BANDS
  prefix = '{PREFIX}'
  outdir = '{OUTDIR}'
  filband = '{label}.bands.dat'
  lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)

    print(f"Wrote {calc_dir}/: pw.in, bands.in, bands_post.in, kpath.json, {PP_FILE}")

print("\nDone. All 51 calculation directories created.")
