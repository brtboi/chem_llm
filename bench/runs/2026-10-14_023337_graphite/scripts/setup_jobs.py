#!/usr/bin/env python3
"""
Setup QE band-structure calculation jobs for graphite C structures.

For each structure NNN (000-050), creates calculations/NNN/ containing:
  - C.pbesol-sr-rrkjus_psl.1.0.0.UPF (pseudopotential, copied from template/)
  - kpath.json (standard k-path from chem_llm.kpath)
  - pw.in (SCF input, written with write_espresso_in)
  - bands.in (bands input, written with write_espresso_in)
  - bands_post.in (bands.x post-processing input)

QE settings (decided in STEP 4):
  - SCF: calculation='scf', prefix='graphite', outdir='./tmp', pseudo_dir='.'
  - ecutwfc = 82.0 Ry (2.0 * 41.0 Ha from Pseudo-Dojo, INVARIANT 5)
  - ecutrho = 328.0 Ry (4 * ecutwfc, standard practice)
  - occupations='smearing', smearing='mv', degauss=0.01 Ry (graphite is semimetal)
  - conv_thr=1e-8
  - nspin=1 (non-magnetic)
  - kpts from scf_kmesh(prim, metal=True)
  - Bands: calculation='bands', same ecutwfc/ecutrho, kpts=kp_kpts(kp)
  - nbnd=20 (z_valence=4, 4 atoms -> 8 electrons -> 4 occupied bands, pad to 20)
  - bands_post.in: &BANDS prefix='graphite', outdir='./tmp', filband='NNN.bands.dat', lsym=.false.
"""

import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# Pseudopotential file
PP_FILE = 'template/C.pbesol-sr-rrkjus_psl.1.0.0.UPF'
PP_NAME = 'C.pbesol-sr-rrkjus_psl.1.0.0.UPF'

# Pseudopotentials dict for write_espresso_in: maps species -> filename
PSEUDOPOTENTIALS = {'C': PP_NAME}

# QE settings
ECUTWFC_RY = 82.0  # 2.0 * 41.0 Ha (INVARIANT 5: 1 Ha = 2 Ry)
ECUTRHO_RY = 328.0  # 4 * ecutwfc
PREFIX = 'graphite'
OUTDIR = './tmp'
CONV_THR = 1e-8
NBND = 20  # 4 occupied bands (8 electrons / 2), padded to 20

# Structure range
N_STRUCTURES = 51

for nnn in range(N_STRUCTURES):
    nnn_str = f'{nnn:03d}'
    calc_dir = f'calculations/{nnn_str}'
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(os.path.join(calc_dir, 'tmp'), exist_ok=True)
    
    # Copy pseudopotential
    shutil.copy(PP_FILE, os.path.join(calc_dir, PP_NAME))
    
    # Read structure
    atoms = ase.io.read(f'structures/structure_{nnn_str}.cif')
    
    # Get standard k-path and primitive cell
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)
    
    # Save k-path JSON
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))
    
    # INVARIANT 1: compute nat and ntyp from the actual Atoms object
    nat = len(prim)
    ntyp = len(set(prim.get_chemical_symbols()))
    assert nat == 4, f"Structure {nnn_str}: expected 4 atoms, got {nat}"
    assert ntyp == 1, f"Structure {nnn_str}: expected 1 element type, got {ntyp}"
    
    # SCF k-point mesh (metal=True for semimetal)
    kpts_scf = scf_kmesh(prim, metal=True)
    
    # SCF input data (FLAT dict, INVARIANT 6)
    scf_input_data = {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'smearing',
        'smearing': 'mv',
        'degauss': 0.01,
        'conv_thr': CONV_THR,
        'nspin': 1,
        'nat': nat,
        'ntyp': ntyp,
    }
    
    # Write SCF input
    # write_espresso_in signature: (fd, atoms, input_data, pseudopotentials, kpts, ...)
    with open(os.path.join(calc_dir, 'pw.in'), 'w') as f:
        write_espresso_in(f, prim, scf_input_data, pseudopotentials=PSEUDOPOTENTIALS, kpts=kpts_scf)
    
    # Bands input data
    bands_input_data = {
        'calculation': 'bands',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'smearing',
        'smearing': 'mv',
        'degauss': 0.01,
        'conv_thr': CONV_THR,
        'nspin': 1,
        'nat': nat,
        'ntyp': ntyp,
        'nbnd': NBND,
    }
    
    # Bands k-points from k-path
    kpts_bands = kp_kpts(kp)
    
    # Write bands input
    with open(os.path.join(calc_dir, 'bands.in'), 'w') as f:
        write_espresso_in(f, prim, bands_input_data, pseudopotentials=PSEUDOPOTENTIALS, kpts=kpts_bands)
    
    # bands_post.in (bands.x input)
    bands_post_content = f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{nnn_str}.bands.dat'
    lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)
    
    if nnn <= 3 or nnn == 50:
        print(f"Setup complete for calculations/{nnn_str}/")

print(f"\nDone. Set up {N_STRUCTURES} calculation directories.")
print("Each contains: pw.in, bands.in, bands_post.in, kpath.json, C.pbesol-sr-rrkjus_psl.1.0.0.UPF")
