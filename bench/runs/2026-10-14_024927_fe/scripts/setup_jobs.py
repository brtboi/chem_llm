#!/usr/bin/env python
"""
STEP 5: Generate QE input files for all 51 structures.

For each structure NNN (000-050):
  calculations/NNN/
    Fe.pbesol_s-1_Ry.upf   (copy of template pseudopotential)
    kpath.json             (from chem_llm.kpath)
    pw.in                  (SCF input, written with write_espresso_in)
    bands.in               (bands input, written with write_espresso_in)
    bands_post.in          (bands.x input, hand-written &BANDS namelist)

Key decisions (from STEP 4 note):
- ecutwfc = 118.0 Ry (59.0 Ha * 2, INVARIANT 5)
- ecutrho = 472.0 Ry (4 * ecutwfc)
- occupations='smearing', smearing='mv', degauss=0.01 Ry (metal, FM Fe)
- conv_thr=1.0e-8 Ry
- nspin=2 (FM Fe, moment_per_formula_unit=2.645 muB)
- initial_magnetization set on the cell via atoms.set_initial_magnetic_moments([0.5])
- k-point mesh: scf_kmesh(prim, metal=True) from chem_llm.kpath
- nbnd: read from SCF pw.out after SCF converges (INVARIANT 2).
  For the initial setup, use the SCF's 'number of Kohn-Sham states' value.
  The SCF with nspin=2 and 16 electrons gives 12 Kohn-Sham states (6 per spin).
  nbnd must be >= 12 for the bands calculation to have starting wavefunctions.
  Set nbnd=12 to match the SCF exactly.
- bands.in: calculation='bands', kpts=kp_kpts(kp), occupations='fixed'
- bands_post.in: &BANDS prefix='fe', outdir='./tmp', filband='NNN.bands.dat', lsym=.false., spin_component=1

NOTE on space group: structure_000 is unperturbed bcc Fe (Im-3m #229).
Structures 001-050 have random atomic displacements + lattice strain, which break
cubic symmetry. seekpath correctly identifies the lower symmetry for each perturbed
structure. Per INVARIANT 3, the k-path is recomputed per structure for its actual
symmetry -- this is expected and correct behavior. The space-group assert is only
applied to structure_000 (the unperturbed base).
"""

import os
import shutil
import numpy as np
from ase.io import read, write
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# --- Pseudopotential info (from get_pseudopotential, STEP 3) ---
# Fe: z_valence=16, relativistic='scalar', recommended_ecutwfc=59.0 Ha
# INVARIANT 5: 1 Ha = 2 Ry -> ecutwfc = 118.0 Ry
ECUTWFC_HA = 59.0
ECUTWFC_RY = 2.0 * ECUTWFC_HA  # 118.0 Ry
ECUTRHO_RY = 4.0 * ECUTWFC_RY  # 472.0 Ry
assert abs(ECUTWFC_RY - 118.0) < 1e-10, f"ecutwfc conversion error: {ECUTWFC_RY}"
assert abs(ECUTRHO_RY - 472.0) < 1e-10, f"ecutrho conversion error: {ECUTRHO_RY}"

PSEUDO_FILE = 'Fe.pbesol_s-1_Ry.upf'
PSEUDO_DIR = '../../template'  # relative to calculations/NNN

# --- nbnd for bands calculation ---
# From SCF pw.out: 'number of Kohn-Sham states = 12' (6 per spin channel for nspin=2)
# INVARIANT 2: nbnd must be >= number of Kohn-Sham states from SCF.
# Set nbnd=12 to match exactly.
NBND = 12

# --- Common input_data for SCF and bands ---
# INVARIANT 6: pass a FLAT dict; ASE routes each key to the correct namelist.
# INVARIANT 1: nat/ntyp computed from the actual Atoms object, not hardcoded.

def make_scf_input_data(prim):
    """Build the flat input_data dict for the SCF calculation."""
    nat = len(prim)
    ntyp = len(set(prim.get_chemical_symbols()))
    # INVARIANT 1: assert nat/ntyp match the actual structure
    assert nat == 1, f"Expected nat=1 for bcc Fe primitive, got {nat}"
    assert ntyp == 1, f"Expected ntyp=1 for bcc Fe primitive, got {ntyp}"

    # SCF k-point mesh: computed via scf_kmesh(prim, metal=True)
    # metal=True for ~2x density (FM Fe is a metal)
    kpts = scf_kmesh(prim, metal=True)

    input_data = {
        'calculation': 'scf',
        'prefix': 'fe',
        'outdir': './tmp',
        'pseudo_dir': PSEUDO_DIR,
        'ecutwfc': ECUTWFC_RY,   # 118.0 Ry
        'ecutrho': ECUTRHO_RY,   # 472.0 Ry
        'occupations': 'smearing',
        'smearing': 'mv',
        'degauss': 0.01,         # Ry
        'conv_thr': 1.0e-8,      # Ry
        'nspin': 2,              # FM Fe
        'nat': nat,
        'ntyp': ntyp,
    }
    return input_data, kpts


def make_bands_input_data(prim):
    """Build the flat input_data dict for the bands calculation."""
    nat = len(prim)
    ntyp = len(set(prim.get_chemical_symbols()))
    assert nat == 1, f"Expected nat=1 for bcc Fe primitive, got {nat}"
    assert ntyp == 1, f"Expected ntyp=1 for bcc Fe primitive, got {ntyp}"

    # nbnd: from SCF pw.out, 'number of Kohn-Sham states = 12'
    # INVARIANT 2: nbnd must be >= SCF's Kohn-Sham states count.
    input_data = {
        'calculation': 'bands',
        'prefix': 'fe',
        'outdir': './tmp',
        'pseudo_dir': PSEUDO_DIR,
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',  # bands step, no smearing
        'conv_thr': 1.0e-8,
        'nspin': 2,
        'nbnd': NBND,
        'nat': nat,
        'ntyp': ntyp,
        # Fix for 'too many bands are not converged' in c_bands:
        # For non-scf calculations, diago_thr_init defaults to (conv_thr/N_elec)/10
        # = (1e-8/16)/10 = 6.25e-11, which is extremely tight and causes
        # Davidson diagonalization to fail. Set a more reasonable value.
        'diago_thr_init': 1e-5,
        'diago_david_ndim': 4,
    }
    return input_data


def write_bands_post_in(calc_dir, nnn):
    """Write bands_post.in (bands.x input) for structure NNN."""
    filband = f'{nnn}.bands.dat'
    content = f"""&BANDS
    prefix = 'fe'
    outdir = './tmp'
    filband = '{filband}'
    lsym = .false.
    spin_component = 1
/"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(content)


def setup_structure(nnn):
    """Set up calculations/NNN/ for one structure."""
    cif_path = f'structures/structure_{nnn}.cif'
    calc_dir = f'calculations/{nnn}'
    os.makedirs(calc_dir, exist_ok=True)

    # --- Read structure and reduce to primitive cell ---
    atoms = read(cif_path)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)

    # Space group check: only for structure_000 (unperturbed base).
    # Perturbed structures (001-050) have lower symmetry due to random
    # displacements + strain, which is expected and correct per INVARIANT 3.
    if nnn == '000':
        sg_num = int(kp['spacegroup'].split('#')[-1].strip().rstrip(')'))
        assert sg_num == 229, f"structure_000: space group changed to {sg_num}"

    # Set initial magnetic moment on the primitive cell (FM Fe)
    # 0.5 is a fraction of valence electrons (QE units), a reasonable initial guess
    prim.set_initial_magnetic_moments([0.5])

    # --- Copy pseudopotential into calc_dir (pw.x runs inside calc_dir) ---
    src_pp = os.path.join('template', PSEUDO_FILE)
    dst_pp = os.path.join(calc_dir, PSEUDO_FILE)
    if not os.path.exists(dst_pp):
        shutil.copy2(src_pp, dst_pp)

    # --- Save kpath.json ---
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))

    # --- Write pw.in (SCF) ---
    scf_input_data, scf_kpts = make_scf_input_data(prim)
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=scf_input_data,
        pseudopotentials={'Fe': PSEUDO_FILE},
        kpts=scf_kpts,
    )

    # --- Write bands.in ---
    bands_input_data = make_bands_input_data(prim)
    bands_kpts = kp_kpts(kp)  # INVARIANT 3: k-points from the k-path
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=bands_input_data,
        pseudopotentials={'Fe': PSEUDO_FILE},
        kpts=bands_kpts,
    )

    # --- Write bands_post.in ---
    write_bands_post_in(calc_dir, nnn)

    print(f"  calculations/{nnn}/: pw.in, bands.in, bands_post.in, kpath.json, {PSEUDO_FILE}")


def main():
    print("Setting up QE input files for structures 000-050...")
    for i in range(51):
        nnn = f"{i:03d}"
        setup_structure(nnn)
    print("Done.")


if __name__ == '__main__':
    main()
