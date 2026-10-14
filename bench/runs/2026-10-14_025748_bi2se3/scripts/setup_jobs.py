#!/usr/bin/env python3
"""
setup_jobs.py -- Generate QE input files for all 51 Bi2Se3 structures.

For each structure NNN (000-050), creates calculations/NNN/ containing:
  - Bi.upf, Se.upf (pseudopotentials copied from template/)
  - kpath.json (from chem_llm.kpath)
  - pw.in (SCF input, written with ase.io.espresso.write_espresso_in)
  - bands.in (NSCF bands input, written with ase.io.espresso.write_espresso_in)
  - bands_post.in (bands.x input, written with write_fortran_namelist)

Key settings (see STEP 4 note):
  - ecutwfc = 104 Ry (52 Ha * 2, from Se pseudopotential, INVARIANT 5)
  - ecutrho = 416 Ry (4 * ecutwfc)
  - noncolin=.true., lspinorb=.true. (spin-orbit, Bi Z=83 > 54)
  - nbnd = 200 (118 occupied with spin-orbit + 82 unoccupied)
  - occupations='fixed' (narrow-gap semiconductor, gap ~0.55 eV)
  - conv_thr = 1e-8 Ry
  - SCF k-mesh from scf_kmesh(prim, metal=False)
"""

import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in, write_fortran_namelist
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# Pseudopotential cutoffs (from get_pseudopotential results, in Ha)
# Bi: 44 Ha, Se: 52 Ha -> max = 52 Ha = 104 Ry (INVARIANT 5: 1 Ha = 2 Ry)
ECUTWFC_HA = 52.0
ECUTWFC_RY = 2.0 * ECUTWFC_HA  # 104 Ry
ECUTRHO_RY = 4.0 * ECUTWFC_RY  # 416 Ry

# Total electrons: 2*Bi(23) + 3*Se(24) = 46 + 72 = 118
# With spin-orbit (noncolin), each band holds 1 electron, so occupied bands = 118
# nbnd = 200 (118 occupied + 82 unoccupied, plenty of margin)
NBND = 200

# Spin-orbit settings (Bi Z=83 > 54, DECISION RULE)
NONCOLIN = True
LSPINORB = True

# Convergence threshold (Ry)
CONV_THR = 1e-8

# Prefix and outdir
PREFIX = 'bi2se3'
OUTDIR = './tmp'

# Pseudopotential directory (relative to calculations/NNN)
# Pseudopotentials are copied into calculations/NNN, so pseudo_dir='.'.
PSEUDO_DIR = '.'

# Pseudopotential files to copy
PSEUDO_FILES = ['Bi.upf', 'Se.upf']

# Pseudopotential mapping for write_espresso_in
# Maps species -> filename (relative to pseudo_dir)
PSEUDOPOTENTIALS = {'Bi': 'Bi.upf', 'Se': 'Se.upf'}

# Number of structures
N_STRUCTURES = 51


def write_bands_post_in(nnn: str, outdir: str = OUTDIR, prefix: str = PREFIX):
    """Write bands.x input file (bands_post.in) for structure NNN.
    
    Uses write_fortran_namelist from ase.io.espresso for proper formatting.
    """
    filband = f'{nnn}.bands.dat'
    
    input_data = {
        'prefix': prefix,
        'outdir': outdir,
        'filband': filband,
        'lsym': False,  # Skip symmetry analysis (slow with spin-orbit)
    }
    
    with open(f'calculations/{nnn}/bands_post.in', 'w') as f:
        write_fortran_namelist(f, input_data, binary='bands')


def setup_structure(nnn: str):
    """Set up QE input files for one structure."""
    calc_dir = f'calculations/{nnn}'
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotentials into the calculation directory
    for ppf in PSEUDO_FILES:
        src = f'template/{ppf}'
        dst = f'{calc_dir}/{ppf}'
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    # Read the structure
    atoms = ase.io.read(f'structures/structure_{nnn}.cif')
    
    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    
    # Get the standard k-path for THIS structure
    # (perturbed structures have lower symmetry and their own path)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in (INVARIANT 3)
    
    # Save kpath.json (holds labels, tick positions, line segments)
    kp_save(kp, f'{calc_dir}/kpath.json')
    
    # SCF k-mesh (Monkhorst-Pack, computed from the primitive cell)
    # metal=False: Bi2Se3 is a narrow-gap semiconductor, not a metal
    kpts_scf = scf_kmesh(prim, metal=False)
    
    # Bands k-points from the k-path (INVARIANT 3)
    kpts_bands = kp_kpts(kp)
    
    # Build input_data for SCF (pw.in)
    # INVARIANT 6: pass a FLAT dict, let ASE route each key to its correct namelist
    scf_input_data = {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': PSEUDO_DIR,
        'ecutwfc': ECUTWFC_RY,   # 104 Ry
        'ecutrho': ECUTRHO_RY,   # 416 Ry
        'occupations': 'fixed',
        'conv_thr': CONV_THR,    # 1e-8 Ry
        'noncolin': NONCOLIN,    # spin-orbit
        'lspinorb': LSPINORB,    # spin-orbit
        # nat and ntyp are written automatically by write_espresso_in from atoms
    }
    
    # Write pw.in (SCF)
    with open(f'{calc_dir}/pw.in', 'w') as f:
        write_espresso_in(f, prim, input_data=scf_input_data, 
                          pseudopotentials=PSEUDOPOTENTIALS, kpts=kpts_scf)
    
    # Build input_data for bands (bands.in)
    bands_input_data = {
        'calculation': 'bands',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': PSEUDO_DIR,
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'noncolin': NONCOLIN,
        'lspinorb': LSPINORB,
        'nbnd': NBND,  # 200 bands (118 occupied + 82 unoccupied)
    }
    
    # Write bands.in (NSCF bands)
    with open(f'{calc_dir}/bands.in', 'w') as f:
        write_espresso_in(f, prim, input_data=bands_input_data, 
                          pseudopotentials=PSEUDOPOTENTIALS, kpts=kpts_bands)
    
    # Write bands_post.in (bands.x)
    write_bands_post_in(nnn)
    
    print(f'Set up calculations/{nnn}/: nat={nat}, ntyp={ntyp}, '
          f'kpts_scf={kpts_scf}, n_kpts_bands={len(kpts_bands)}')


def main():
    # Create calculations directory
    os.makedirs('calculations', exist_ok=True)
    
    for i in range(N_STRUCTURES):
        nnn = f'{i:03d}'
        setup_structure(nnn)
    
    print(f'\nDone. Set up {N_STRUCTURES} calculation directories in calculations/.')


if __name__ == '__main__':
    main()
