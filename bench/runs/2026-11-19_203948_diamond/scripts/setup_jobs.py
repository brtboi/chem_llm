#!/usr/bin/env python3
"""STEP 5: Generate QE input files for all 51 diamond-C structures.

For each structure NNN:
  calculations/NNN/
    C.upf          (copied from template/)
    kpath.json     (MP's path for 000, seekpath for perturbed)
    pw.in          (SCF input, written by write_espresso_in)
    bands.in       (NSCF bands input, written by write_espresso_in)
    bands_post.in  (bands.x input, hand-written -- no ASE writer for bands.x)

Key decisions (see STEP 4 note):
  ecutwfc = 30.0 Ry  (= 15.0 Ha x 2, INVARIANT 5)
  ecutrho = 240.0 Ry (= 8 x ecutwfc, PAW requirement)
  conv_thr = 1.0e-8 Ry
  occupations = 'fixed' (insulator, gap ~4.1 eV)
  nspin = 1 (no magnetism)
  No spin-orbit (matches MP's VASP run)
  No Hubbard U
  nbnd = 16 (4 occupied bands, padded well above)
  SCF k-mesh: scf_kmesh(prim, metal=False, spacing=0.25) -- converged by converge_kmesh
  Bands k-points: kp_kpts(kp) from kpath.json (241 k-points for 000)
"""
import os
import shutil
import numpy as np
from ase.io import read, write
from ase.io.espresso import write_espresso_in

# --- chem_llm.dft.kpath functions (same code as get_kpath tool) ---
from chem_llm.dft.kpath import (
    standard_kpath, kp_atoms, kp_kpts, kp_save, load_kpath, scf_kmesh,
)

# --- Constants from STEP 3/4 ---
ECUTWFC_RY = 30.0   # 15.0 Ha * 2 (INVARIANT 5: 1 Ha = 2 Ry)
ECUTRHO_RY = 240.0  # 8 * ecutwfc (PAW)
CONV_THR = 1.0e-8   # Ry
NBND = 16           # 4 occupied bands (8 e- / 2), padded
PREFIX = 'diamond'
OUTDIR = './tmp'
# pseudo_dir: relative to calculations/NNN; template/ is at project root
# calculations/NNN is two levels below project root, so '../../template'
PSEUDO_DIR = '../../template'

N_STRUCTURES = 51

# SCF k-mesh spacing (Angstrom^-1) from converge_kmesh result.
# converge_kmesh returned spacing=0.25, mesh=[13,13,13].
SCF_SPACING = 0.25


def make_pw_in(prim, nnn, kpts):
    """Write pw.in (SCF) for structure NNN using write_espresso_in."""
    nat = len(prim)
    ntyp = len(set(prim.get_chemical_symbols()))
    # INVARIANT 1: nat/ntyp computed from the actual Atoms object
    assert nat == 2, f'pw.in {nnn}: expected 2 atoms, got {nat}'
    assert ntyp == 1, f'pw.in {nnn}: expected 1 species, got {ntyp}'

    input_data = {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': PSEUDO_DIR,
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'conv_thr': CONV_THR,
        'occupations': 'fixed',
        'nspin': 1,
    }
    # write_espresso_in routes each key to the correct namelist (INVARIANT 6)
    with open(f'calculations/{nnn}/pw.in', 'w') as f:
        write_espresso_in(f, prim, input_data=input_data,
                          pseudopotentials={'C': 'C.upf'},
                          kpts=kpts)


def make_bands_in(prim, nnn, kpts):
    """Write bands.in (NSCF bands) for structure NNN using write_espresso_in."""
    nat = len(prim)
    ntyp = len(set(prim.get_chemical_symbols()))
    assert nat == 2, f'bands.in {nnn}: expected 2 atoms, got {nat}'
    assert ntyp == 1, f'bands.in {nnn}: expected 1 species, got {ntyp}'

    input_data = {
        'calculation': 'bands',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': PSEUDO_DIR,
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'conv_thr': CONV_THR,
        'occupations': 'fixed',
        'nspin': 1,
        'nbnd': NBND,
    }
    with open(f'calculations/{nnn}/bands.in', 'w') as f:
        write_espresso_in(f, prim, input_data=input_data,
                          pseudopotentials={'C': 'C.upf'},
                          kpts=kpts)


def make_bands_post_in(nnn):
    """Write bands_post.in (bands.x input) for structure NNN.

    bands.x has no ASE writer, so we write it directly.
    &BANDS namelist: prefix, outdir, filband, lsym=.false.
    lsym=.false. skips the band-representation file (not needed, slow with SO).
    spin_component=1 (only component for nspin=1).
    """
    content = f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{nnn}.bands.dat'
    lsym = .false.
    spin_component = 1
/
"""
    with open(f'calculations/{nnn}/bands_post.in', 'w') as f:
        f.write(content)


def main():
    # Load MP's k-path for structure 000 (unperturbed cell only)
    kp_000 = load_kpath('structures/kpath.json')
    # kpath.json has 'path' and 'kpoints' keys (not 'n_kpoints')
    n_kpts_000 = len(kp_000['kpoints'])
    print(f'MP k-path: {kp_000["path"]}, {n_kpts_000} k-points')

    for nnn in range(N_STRUCTURES):
        nnn_str = f'{nnn:03d}'
        calc_dir = f'calculations/{nnn_str}'
        os.makedirs(calc_dir, exist_ok=True)

        # Copy pseudopotential into the calculation directory
        # (pw.x runs inside calculations/NNN, so pseudo_dir must resolve from there)
        shutil.copy('template/C.upf', f'{calc_dir}/C.upf')

        # Read the structure
        cif_path = f'structures/structure_{nnn_str}.cif'
        atoms = read(cif_path)

        # Get the k-path and the cell it refers to
        if nnn == 0:
            # Structure 000: use MP's k-path (belongs to the unperturbed cell)
            kp = kp_000
        else:
            # Perturbed structures: lower symmetry, use seekpath's standard path
            kp = standard_kpath(atoms)

        # kp_atoms(kp) is THE cell for pw.in AND bands.in (INVARIANT 3)
        prim = kp_atoms(kp)

        # Save kpath.json for this structure
        kp_save(kp, f'{calc_dir}/kpath.json')

        # SCF k-mesh: computed by scf_kmesh for the cell the SCF runs on
        # spacing=0.25 from converge_kmesh (STEP 6)
        kpts_scf = scf_kmesh(prim, metal=False, spacing=SCF_SPACING)
        print(f'  {nnn_str}: SCF k-mesh = {kpts_scf}')

        # Bands k-points: from kpath.json (INVARIANT 3)
        kpts_bands = kp_kpts(kp)

        # Write input files
        make_pw_in(prim, nnn_str, kpts_scf)
        make_bands_in(prim, nnn_str, kpts_bands)
        make_bands_post_in(nnn_str)

        print(f'  {nnn_str}: done ({len(prim)} atoms, {len(kpts_bands)} bands kpts)')

    print(f'\nAll {N_STRUCTURES} calculation directories written.')


if __name__ == '__main__':
    main()
