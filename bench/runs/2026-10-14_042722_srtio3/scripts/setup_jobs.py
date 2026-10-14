#!/usr/bin/env python
"""
setup_jobs.py -- Generate QE input files for all 51 SrTiO3 structures.

For each structure NNN (000-050), creates calculations/NNN/ containing:
  - Pseudopotential files (copied from template/)
  - kpath.json (from chem_llm.kpath)
  - pw.in (SCF input, written with write_espresso_in)
  - bands.in (NSCF bands input, written with write_espresso_in)
  - bands_post.in (bands.x post-processing input)

Key settings (see STEP 4 note):
  - ecutwfc = 92 Ry (from max recommended cutoff 46 Ha * 2 Ry/Ha, INVARIANT 5)
  - ecutrho = 368 Ry (4 * ecutwfc)
  - occupations = 'fixed' (semiconductor, gap ~1.77 eV)
  - conv_thr = 1.0d-8 (in &ELECTRONS, INVARIANT 6)
  - nspin = 1 (no magnetism)
  - nbnd = 40 (2x occupied bands = 2 * 20, where 20 = 40 electrons / 2)
  - SCF k-mesh: computed via scf_kmesh(prim, metal=False)
  - Bands k-points: from kp_kpts(kp) (INVARIANT 3)
"""

import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# --- Constants ---
N_STRUCTURES = 51  # 000 through 050
TEMPLATE_DIR = 'template'
STRUCTURES_DIR = 'structures'
CALC_DIR = 'calculations'

# Pseudopotential files (from get_pseudopotential results)
PSEUDO_FILES = {
    'Sr': 'Sr.pbesol-sr-rrkjus_psl.1.0.0.UPF',
    'Ti': 'Ti.pbesol-sr-rrkjus_psl.1.0.0.UPF',
    'O': 'O.pbesol-sr-rrkjus_psl.1.0.0.UPF',
}

# Cutoffs: max recommended cutoff is 46 Ha (O) -> 46 * 2 = 92 Ry (INVARIANT 5)
ECUTWFC_RY = 92.0
ECUTRHO_RY = 4.0 * ECUTWFC_RY  # 368 Ry

# Occupied bands: total electrons = 10(Sr) + 12(Ti) + 3*6(O) = 40
# Occupied bands = 40/2 = 20 (spin-unpolarized)
# nbnd = 2 * occupied = 40 (pad for conduction bands)
NBND = 40

# conv_thr in Ry (QE default unit)
CONV_THR = 1.0e-8

# prefix and outdir
PREFIX = 'srtio3'
OUTDIR = './tmp'


def setup_one_structure(nnn: str):
    """Set up calculations/NNN/ for structure NNN."""
    calc_dir = os.path.join(CALC_DIR, nnn)
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(os.path.join(calc_dir, OUTDIR), exist_ok=True)

    # 1. Copy pseudopotential files into calc_dir
    for elem, fname in PSEUDO_FILES.items():
        src = os.path.join(TEMPLATE_DIR, fname)
        dst = os.path.join(calc_dir, fname)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Pseudopotential not found: {src}")
        shutil.copy2(src, dst)

    # 2. Read the structure
    cif_path = os.path.join(STRUCTURES_DIR, f'structure_{nnn}.cif')
    atoms = ase.io.read(cif_path)

    # 3. Get k-path and primitive cell (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in

    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(prim)
    ntyp = len(set(prim.get_chemical_symbols()))
    assert nat == 5, f"Structure {nnn}: expected nat=5, got {nat}"
    assert ntyp == 3, f"Structure {nnn}: expected ntyp=3, got {ntyp}"

    # 4. Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))

    # 5. SCF k-mesh (Monkhorst-Pack, computed from the cell)
    scf_kpts = scf_kmesh(prim, metal=False)

    # 6. Bands k-points (from kpath)
    bands_kpts = kp_kpts(kp)

    # 7. Pseudopotential dict for write_espresso_in
    pseudos = {elem: fname for elem, fname in PSEUDO_FILES.items()}

    # 8. Write pw.in (SCF)
    scf_input_data = {
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
        input_data=scf_input_data,
        pseudopotentials=pseudos,
        kpts=scf_kpts,
    )

    # 9. Write bands.in (NSCF bands)
    bands_input_data = {
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
        input_data=bands_input_data,
        pseudopotentials=pseudos,
        kpts=bands_kpts,
    )

    # 10. Write bands_post.in (bands.x input)
    #     lsym=.false. to skip symmetry analysis (saves ~20 min)
    bands_post_content = f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{nnn}.bands.dat'
    lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)

    print(f"  {nnn}: nat={nat}, ntyp={ntyp}, scf_kpts={scf_kpts}, nbnd={NBND}")


def main():
    print("Setting up QE input files for all 51 structures...")
    for n in range(N_STRUCTURES):
        nnn = f"{n:03d}"
        setup_one_structure(nnn)
    print(f"\nDone. Set up {N_STRUCTURES} calculation directories in {CALC_DIR}/")


if __name__ == '__main__':
    main()
