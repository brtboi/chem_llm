#!/usr/bin/env python3
"""
setup_jobs.py - Generate QE input files for all 51 PbTe structures.

For each structure NNN (000-050), creates calculations/NNN/ containing:
  - Pb.upf, Te.upf (pseudopotentials copied from template/)
  - kpath.json (from chem_llm.kpath)
  - pw.in (SCF input)
  - bands.in (NSCF bands input)
  - bands_post.in (bands.x post-processing input)

Key settings (see STEP 4 note):
  - ecutwfc = 102 Ry (51 Ha * 2, from Te pseudopotential, INVARIANT 5)
  - ecutrho = 408 Ry (4 * ecutwfc)
  - noncolin = .true., lspinorb = .true. (spin-orbit, Pb Z=82 > 54)
  - nbnd = 60 (46 electrons with SO, each band holds 1 electron, padding)
  - occupations = 'fixed' (PbTe is a semiconductor, gap ~0.8 eV)
  - conv_thr = 1e-8
  - SCF k-mesh from scf_kmesh(prim, metal=False)
  - bands.in kpts from kp_kpts(kp)
  - bands_post.in: lsym=.false. (skip symmetry analysis), spin_component=1
"""

import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# Pseudopotential files
PSEUDO_DIR = 'template'
PSEUDOS = {'Pb': 'Pb.upf', 'Te': 'Te.upf'}

# Cutoffs (INVARIANT 5: convert from Ha to Ry, 1 Ha = 2 Ry)
# Te: 51 Ha = 102 Ry (larger of the two)
# Pb: 44 Ha = 88 Ry
ECUTWFC_RY = 102.0  # Ry
ECUTRHO_RY = 4 * ECUTWFC_RY  # 408 Ry

# Spin-orbit settings (Pb Z=82 > 54, fully relativistic pseudopotentials)
NONCOLIN = True
LSPINORB = True

# Number of bands: 46 electrons (22 Pb + 24 Te), with SO each band holds 1 electron
# So occupied bands = 46, nbnd = 60 for padding
NBND = 60

# Convergence threshold
CONV_THR = 1e-8

# Prefix and outdir
PREFIX = 'pbte'
OUTDIR = './tmp'


def setup_structure(nnn: str, base_atoms):
    """Set up QE input files for one structure."""
    calc_dir = f'calculations/{nnn}'
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(os.path.join(calc_dir, OUTDIR), exist_ok=True)

    # Copy pseudopotentials into the calculation directory
    for elem, fname in PSEUDOS.items():
        src = os.path.join(PSEUDO_DIR, fname)
        dst = os.path.join(calc_dir, fname)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Read the structure
    cif_path = f'structures/structure_{nnn}.cif'
    atoms = ase.io.read(cif_path)

    # Get the k-path and primitive cell (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)

    # Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))

    # INVARIANT 1: nat and ntyp from the actual structure
    nat = len(prim)
    ntyp = len(set(prim.get_chemical_symbols()))
    assert nat == 2, f"Structure {nnn}: expected 2 atoms, got {nat}"
    assert ntyp == 2, f"Structure {nnn}: expected 2 species, got {ntyp}"

    # SCF k-mesh (metal=False since PbTe is a semiconductor)
    kpts_scf = scf_kmesh(prim, metal=False)

    # Build input_data for pw.in (SCF)
    input_data_scf = {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
        'noncolin': NONCOLIN,
        'lspinorb': LSPINORB,
    }

    # Write pw.in
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=input_data_scf,
        pseudopotentials=PSEUDOS,
        kpts=kpts_scf,
    )

    # Build input_data for bands.in (NSCF bands)
    input_data_bands = {
        'calculation': 'bands',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
        'noncolin': NONCOLIN,
        'lspinorb': LSPINORB,
        'nbnd': NBND,
    }

    # Write bands.in with kpts from the k-path
    kpts_bands = kp_kpts(kp)
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=input_data_bands,
        pseudopotentials=PSEUDOS,
        kpts=kpts_bands,
    )

    # Write bands_post.in (bands.x input)
    # lsym=.false. to skip symmetry analysis (saves ~20 min with spin-orbit)
    # spin_component=1 for majority spin in noncolin case
    bands_post_content = f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{nnn}.bands.dat'
    lsym = .false.
    spin_component = 1
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)

    print(f"Setup complete for {nnn}: {nat} atoms, {ntyp} species, "
          f"SCF kpts={kpts_scf}, bands kpts={len(kpts_bands)} points")


def main():
    # Read the base structure to get the cell (for reference)
    base = ase.io.read('structures/structure_000.cif')
    print(f"Base structure: {len(base)} atoms, cell: {base.get_cell()}")

    # Set up all 51 structures
    for i in range(51):
        nnn = f"{i:03d}"
        setup_structure(nnn, base)

    print("\nAll 51 calculation directories set up.")
    print("Run validate_qe_input on calculations/000/pw.in and calculations/000/bands.in")


if __name__ == '__main__':
    main()
