#!/usr/bin/env python
"""STEP 5: Generate QE input files for all 51 Si structures.

For each structure NNN (000-050), creates calculations/NNN/ containing:
  - Si.upf (pseudopotential, copied from template/)
  - kpath.json (k-path from chem_llm.dft.kpath)
  - pw.in (SCF input, written with write_espresso_in)
  - bands.in (NSCF bands input, written with write_espresso_in)
  - bands_post.in (bands.x post-processing input)

Key decisions (documented in STEP 4 note):
  - Pseudopotential: Si.upf (PAW, scalar-relativistic, PBE, stringent)
  - ecutwfc = 20.0 Ry (= 10.0 Ha, Pseudo-Dojo recommended; INVARIANT 5: 1 Ha = 2 Ry)
  - ecutrho = 160.0 Ry (= 8 * ecutwfc, required for PAW pseudopotentials)
  - occupations = 'fixed' (Si is a clear semiconductor, ~1.1 eV gap)
  - conv_thr = 1.0e-8 Ry
  - kpts: computed by scf_kmesh(prim, spacing=0.2) for SCF (converged by converge_kmesh)
  - nbnd = 16 in bands.in (>= 4 occupied bands, generous for conduction bands)
  - prefix = 'si', outdir = './tmp', pseudo_dir = '.'
  - bands_post.in: filband = 'NNN.bands.dat', lsym = .false.
"""

import os
import shutil
import numpy as np
from ase.io import read, write
from ase.io.espresso import write_espresso_in
from chem_llm.dft.kpath import (
    standard_kpath, kp_atoms, kp_kpts, kp_save, load_kpath, scf_kmesh
)

# --- Constants ---
ECUTWFC_RY = 20.0   # Ry (= 10.0 Ha, Pseudo-Dojo recommended cutoff for Si.upf)
ECUTRHO_RY = 160.0  # Ry (= 8 * ecutwfc, required for PAW pseudopotentials)
CONV_THR = 1.0e-8   # Ry, SCF convergence threshold
NBND = 16           # bands: >= 4 occupied bands (8 e- / 2), generous for conduction
PREFIX = 'si'
OUTDIR = './tmp'
PSEUDO_DIR = '.'

# INVARIANT 5: verify the unit conversion
assert abs(ECUTWFC_RY - 2.0 * 10.0) < 1e-12, \
    f"ecutwfc {ECUTWFC_RY} Ry != 2 * 10.0 Ha"
assert abs(ECUTRHO_RY - 8.0 * ECUTWFC_RY) < 1e-12, \
    f"ecutrho {ECUTRHO_RY} Ry != 8 * ecutwfc {ECUTWFC_RY} Ry"

# --- Pseudopotential file ---
PP_FILE = 'Si.upf'
PP_SRC = os.path.join('template', PP_FILE)
assert os.path.exists(PP_SRC), f"Pseudopotential not found: {PP_SRC}"

# --- SCF input_data (flat dict; ASE routes each key to the correct namelist) ---
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

# --- Bands input_data (flat dict) ---
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

# --- bands_post.in template (bands.x input) ---
def make_bands_post_in(nnn: str) -> str:
    """Write the bands.x post-processing input for structure NNN."""
    return f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{nnn}.bands.dat'
    lsym = .false.
    spin_component = 1
/
"""

# --- Main loop ---
for i in range(51):
    nnn = f"{i:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)

    # Copy pseudopotential into the calculation directory
    pp_dst = os.path.join(calc_dir, PP_FILE)
    if not os.path.exists(pp_dst):
        shutil.copy2(PP_SRC, pp_dst)

    # Read the structure
    cif_path = f"structures/structure_{nnn}.cif"
    atoms = read(cif_path)

    # INVARIANT 1: nat/ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Structure {nnn}: nat {nat} != 2"
    assert ntyp == 1, f"Structure {nnn}: ntyp {ntyp} != 1"

    # K-path: for structure 000 use MP's path (load_kpath),
    # for perturbed structures use standard_kpath (seekpath)
    if i == 0:
        kp = load_kpath('structures/kpath.json')
    else:
        kp = standard_kpath(atoms)

    # The cell for pw.in and bands.in is kp_atoms(kp) -- INVARIANT 3
    prim = kp_atoms(kp)

    # INVARIANT 1: re-check nat/ntyp on the cell we actually write
    nat_prim = len(prim)
    ntyp_prim = len(set(prim.get_chemical_symbols()))
    assert nat_prim == nat, f"Structure {nnn}: prim nat {nat_prim} != {nat}"
    assert ntyp_prim == ntyp, f"Structure {nnn}: prim ntyp {ntyp_prim} != {ntyp}"

    # Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))

    # SCF k-mesh: computed by scf_kmesh with the converged spacing (0.2)
    # from converge_kmesh in STEP 6
    kpts_scf = scf_kmesh(prim, spacing=0.2)

    # Write pw.in (SCF)
    pw_in_path = os.path.join(calc_dir, 'pw.in')
    write_espresso_in(
        pw_in_path,
        prim,
        input_data=scf_input_data,
        pseudopotentials={'Si': PP_FILE},
        kpts=kpts_scf,
    )

    # Write bands.in (NSCF bands)
    bands_in_path = os.path.join(calc_dir, 'bands.in')
    kpts_bands = kp_kpts(kp)
    write_espresso_in(
        bands_in_path,
        prim,
        input_data=bands_input_data,
        pseudopotentials={'Si': PP_FILE},
        kpts=kpts_bands,
    )

    # Write bands_post.in (bands.x)
    bands_post_path = os.path.join(calc_dir, 'bands_post.in')
    with open(bands_post_path, 'w') as f:
        f.write(make_bands_post_in(nnn))

    if i < 3 or i == 50:
        print(f"Structure {nnn}: nat={nat}, ntyp={ntyp}, "
              f"kpts_scf={kpts_scf}, n_kpts_bands={len(kpts_bands)}")

print("Done: wrote QE inputs for all 51 structures.")
