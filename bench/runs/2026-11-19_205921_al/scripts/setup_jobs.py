#!/usr/bin/env python
"""
STEP 5: Write setup_jobs.py -- create calculations/NNN/ for each structure with
QE input files (pw.in, bands.in, bands_post.in) and pseudopotential files.

Key decisions (from STEP 4 note):
- ecutwfc = 24.0 Ry (12.0 Ha x 2, INVARIANT 5)
- ecutrho = 192.0 Ry (8 x ecutwfc for PAW)
- occupations='smearing', smearing='mv', degauss=0.01 Ry (Al is a metal)
- conv_thr=1e-8
- nbnd=8 (3 electrons -> 1.5 occupied bands, padded)
- No spin-orbit, no Hubbard U, nspin=1
- For structure 000: load_kpath('structures/kpath.json') (MP's path)
- For perturbed structures: standard_kpath(atoms)
- SCF k-mesh: scf_kmesh(prim, spacing=0.1024) (converged by converge_kmesh)
"""

import os
import shutil
import numpy as np
from ase.io import read, write
from ase.io.espresso import write_espresso_in

# --- Configuration ---
N_STRUCTURES = 51
STRUCTURES_DIR = 'structures'
CALC_DIR = 'calculations'
TEMPLATE_DIR = 'template'

# Pseudopotential file (copied into each calculations/NNN/)
PSEUDO_FILE = 'Al.upf'

# QE settings (from STEP 4 note)
ECUTWFC_RY = 24.0   # 12.0 Ha * 2 (INVARIANT 5)
ECUTRHO_RY = 192.0  # 8 * ecutwfc for PAW
DEGAUSS_RY = 0.01   # smearing width for metal
CONV_THR = 1e-8     # SCF convergence threshold (Ry)
NBND = 8            # number of bands (padded above 1.5 occupied)
PREFIX = 'al'
OUTDIR = './tmp'

# K-mesh spacing (converged by converge_kmesh in STEP 6)
# converge_kmesh returned spacing=0.1024, mesh=[27,27,27]
K_SPACING = 0.1024


def get_kpath_for_structure(nnn, atoms):
    """
    Get the k-path for a structure.
    
    For structure 000 (unperturbed base): use MP's k-path from structures/kpath.json
    For perturbed structures: use standard_kpath(atoms) (seekpath)
    """
    from chem_llm.dft.kpath import load_kpath, standard_kpath
    
    if nnn == '000':
        # MP's k-path, which belongs to the unperturbed cell only
        kp = load_kpath(os.path.join(STRUCTURES_DIR, 'kpath.json'))
    else:
        # Perturbed structure: recompute k-path (lower symmetry)
        kp = standard_kpath(atoms)
    
    return kp


def write_bands_post_in(path, nnn):
    """
    Write the bands.x post-processing input file.
    
    &BANDS namelist:
    - prefix: same as pw.x
    - outdir: same as pw.x
    - filband: output file name for band data
    - lsym: .false. (skip band-representation file, saves time)
    """
    content = f"""&BANDS
    prefix = '{PREFIX}',
    outdir = '{OUTDIR}',
    filband = '{nnn}.bands.dat',
    lsym = .false.
/
"""
    with open(path, 'w') as f:
        f.write(content)


def main():
    os.makedirs(CALC_DIR, exist_ok=True)
    
    # Copy pseudopotential to template (already there, but ensure it exists)
    pseudo_src = os.path.join(TEMPLATE_DIR, PSEUDO_FILE)
    assert os.path.exists(pseudo_src), f"Pseudopotential not found: {pseudo_src}"
    
    for n in range(N_STRUCTURES):
        nnn = f"{n:03d}"
        calc_path = os.path.join(CALC_DIR, nnn)
        os.makedirs(calc_path, exist_ok=True)
        
        # Copy pseudopotential into calculations/NNN/
        pseudo_dst = os.path.join(calc_path, PSEUDO_FILE)
        if not os.path.exists(pseudo_dst):
            shutil.copy2(pseudo_src, pseudo_dst)
        
        # Read structure
        cif_path = os.path.join(STRUCTURES_DIR, f"structure_{nnn}.cif")
        atoms = read(cif_path)
        
        # INVARIANT 1: compute nat and ntyp from the actual Atoms object
        nat = len(atoms)
        ntyp = len(set(atoms.get_chemical_symbols()))
        assert nat == 1, f"Expected nat=1 for Al primitive cell, got {nat}"
        assert ntyp == 1, f"Expected ntyp=1 for pure Al, got {ntyp}"
        
        # Get k-path
        kp = get_kpath_for_structure(nnn, atoms)
        
        # Get the primitive cell for the k-path (INVARIANT 3)
        from chem_llm.dft.kpath import kp_atoms, kp_kpts, kp_save, scf_kmesh
        prim = kp_atoms(kp)
        
        # Save k-path JSON
        kp_save(kp, os.path.join(calc_path, 'kpath.json'))
        
        # SCF k-mesh (converged spacing from converge_kmesh)
        kpts = scf_kmesh(prim, spacing=K_SPACING)
        
        # Pseudopotential mapping
        pseudos = {'Al': PSEUDO_FILE}
        
        # --- Write pw.in (SCF) ---
        scf_input_data = {
            'calculation': 'scf',
            'prefix': PREFIX,
            'outdir': OUTDIR,
            'pseudo_dir': './',
            'ecutwfc': ECUTWFC_RY,
            'ecutrho': ECUTRHO_RY,
            'occupations': 'smearing',
            'smearing': 'mv',
            'degauss': DEGAUSS_RY,
            'conv_thr': CONV_THR,
            'nspin': 1,
        }
        
        write_espresso_in(
            os.path.join(calc_path, 'pw.in'),
            prim,
            input_data=scf_input_data,
            pseudopotentials=pseudos,
            kpts=kpts,
        )
        
        # --- Write bands.in (NSCF bands) ---
        bands_input_data = {
            'calculation': 'bands',
            'prefix': PREFIX,
            'outdir': OUTDIR,
            'pseudo_dir': './',
            'ecutwfc': ECUTWFC_RY,
            'ecutrho': ECUTRHO_RY,
            'occupations': 'fixed',
            'nbnd': NBND,
            'nspin': 1,
        }
        
        write_espresso_in(
            os.path.join(calc_path, 'bands.in'),
            prim,
            input_data=bands_input_data,
            pseudopotentials=pseudos,
            kpts=kp_kpts(kp),
        )
        
        # --- Write bands_post.in (bands.x) ---
        write_bands_post_in(os.path.join(calc_path, 'bands_post.in'), nnn)
        
        if n == 0 or n % 10 == 0:
            print(f"  Wrote {calc_path}/ (nat={nat}, ntyp={ntyp}, kpts={kpts})")
    
    print(f"\nDone. Wrote {N_STRUCTURES} calculation directories to {CALC_DIR}/")
    print(f"ecutwfc={ECUTWFC_RY} Ry, ecutrho={ECUTRHO_RY} Ry, nbnd={NBND}")
    print(f"K_SPACING={K_SPACING} (converged by converge_kmesh)")


if __name__ == '__main__':
    main()
