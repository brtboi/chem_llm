"""
STEP 5: Write setup_jobs.py -- generates QE input files for all 51 Al structures.

For each structure NNN (000-050):
  - Create calculations/NNN/
  - Copy Al.upf into calculations/NNN/
  - Get k-path: for 000, load MP's path from structures/kpath.json;
    for perturbed structures, use standard_kpath(atoms)
  - Write pw.in (SCF) and bands.in (NSCF bands) with write_espresso_in
  - Write bands_post.in (bands.x input) with write_fortran_namelist

Key decisions (from STEP 4 note):
  - ecutwfc = 30.0 Ry (15.0 Ha * 2, INVARIANT 5)
  - ecutrho = 240.0 Ry (8 * ecutwfc for PAW)
  - occupations='smearing', smearing='mv', degauss=0.01 Ry (Al is a metal)
  - conv_thr=1.0e-8 Ry
  - nspin=1 (no magnetism)
  - No spin-orbit (matching MP)
  - nbnd=16 for bands (3 electrons -> 2 occupied bands, need > 2)
  - SCF k-mesh: scf_kmesh(prim, spacing=0.1024) -- converged by converge_kmesh
  - Bands k-points: kp_kpts(kp) (MP's k-path for 000, standard for others)
"""

import os
import shutil
import numpy as np
from ase.io import read
from ase.io.espresso import write_espresso_in, write_fortran_namelist
from chem_llm.dft.kpath import (
    standard_kpath, kp_atoms, kp_kpts, kp_save, load_kpath, scf_kmesh
)

# --- Pseudopotential ---
# Al PAW PBE stringent UPF, scalar-relativistic, no spin-orbit
# z_valence = 3.0 (matches MP's PAW_PBE Al 04Jan2001 standard valence)
# Recommended ecutwfc = 15.0 Ha = 30.0 Ry (INVARIANT 5: 1 Ha = 2 Ry)
# PAW: ecutrho = 8 * ecutwfc = 240.0 Ry
PSEUDO_DIR = 'template'
AL_PSEUDO = 'Al.upf'
ECUTWFC_RY = 30.0  # Ry (15.0 Ha * 2)
ECUTRHO_RY = 240.0  # Ry (8 * 30.0 for PAW)

# --- QE settings ---
PREFIX = 'al'
OUTDIR = './tmp'
CONV_THR = 1.0e-8  # Ry
N_SPIN = 1  # no magnetism
NBND = 16  # 3 electrons -> 2 occupied bands, need > 2; 16 covers conduction bands

# Al is a metal (band_gap_eV = 0.0 from MP)
OCCUPATIONS = 'smearing'
SMEARING = 'mv'
DEGAUSS = 0.01  # Ry

N_STRUCTURES = 51

# --- SCF k-mesh spacing from converge_kmesh (STEP 6) ---
# converge_kmesh returned spacing=0.1024, mesh [27,27,27]
SCF_SPACING = 0.1024

for i in range(N_STRUCTURES):
    nnn = f'{i:03d}'
    calc_dir = f'calculations/{nnn}'
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(os.path.join(calc_dir, 'tmp'), exist_ok=True)
    
    # Copy pseudopotential into the calculation directory
    shutil.copy(os.path.join(PSEUDO_DIR, AL_PSEUDO), calc_dir)
    
    # Read the structure
    atoms = read(f'structures/structure_{nnn}.cif')
    
    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 1, f"Structure {nnn}: nat={nat}, expected 1"
    assert ntyp == 1, f"Structure {nnn}: ntyp={ntyp}, expected 1"
    
    # Get the k-path
    if i == 0:
        # Structure 000: use MP's k-path (belongs to the unperturbed cell)
        kp = load_kpath('structures/kpath.json')
    else:
        # Perturbed structures: use standard_kpath (seekpath)
        kp = standard_kpath(atoms)
    
    # The cell for pw.in and bands.in is kp_atoms(kp)
    # (INVARIANT 3: the k-path's fractional k-points refer to this cell's reciprocal basis)
    prim = kp_atoms(kp)
    
    # Save the k-path to the calculation directory
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))
    
    # --- SCF k-mesh ---
    # Use the converged spacing from converge_kmesh (STEP 6)
    kpts_scf = scf_kmesh(prim, spacing=SCF_SPACING)
    
    # --- Bands k-points ---
    # For the bands calculation, use the k-path's k-points
    kpts_bands = kp_kpts(kp)
    
    # --- Pseudopotential mapping ---
    # write_espresso_in expects a dict mapping species -> filename
    # The filename is relative to pseudo_dir
    pseudos = {'Al': AL_PSEUDO}
    
    # --- SCF input (pw.in) ---
    input_data_scf = {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',  # pseudopotentials are in the same directory
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': OCCUPATIONS,
        'smearing': SMEARING,
        'degauss': DEGAUSS,
        'conv_thr': CONV_THR,
        'nspin': N_SPIN,
    }
    
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=input_data_scf,
        pseudopotentials=pseudos,
        kpts=kpts_scf,
    )
    
    # --- Bands input (bands.in) ---
    input_data_bands = {
        'calculation': 'bands',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': OCCUPATIONS,
        'smearing': SMEARING,
        'degauss': DEGAUSS,
        'conv_thr': CONV_THR,
        'nspin': N_SPIN,
        'nbnd': NBND,
    }
    
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=input_data_bands,
        pseudopotentials=pseudos,
        kpts=kpts_bands,
    )
    
    # --- Bands post-processing input (bands_post.in) ---
    # bands.x input with filband = 'NNN.bands.dat' and lsym = .false.
    # lsym=.false. because the symmetry analysis only writes a band-representation
    # file nothing here uses, and with spin-orbit it alone has taken 20 minutes.
    bands_post_data = {
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'filband': f'{nnn}.bands.dat',
        'lsym': False,
    }
    
    write_fortran_namelist(
        os.path.join(calc_dir, 'bands_post.in'),
        input_data=bands_post_data,
        binary='bands',
    )
    
    print(f"Wrote {calc_dir}/: pw.in, bands.in, bands_post.in, kpath.json, {AL_PSEUDO}")

print(f"\nDone: wrote QE inputs for {N_STRUCTURES} structures.")
print(f"ecutwfc={ECUTWFC_RY} Ry, ecutrho={ECUTRHO_RY} Ry, nbnd={NBND}")
print(f"occupations={OCCUPATIONS}, smearing={SMEARING}, degauss={DEGAUSS} Ry")
print(f"SCF k-mesh spacing: {SCF_SPACING}")
