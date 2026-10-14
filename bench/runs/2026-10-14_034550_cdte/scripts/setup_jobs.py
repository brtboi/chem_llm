#!/usr/bin/env python
"""
Setup QE input files for all 51 CdTe structures.

For each structure NNN (000-050), create calculations/NNN/ containing:
- Pseudopotential files (copied from template/)
- kpath.json (from chem_llm.kpath)
- pw.in (SCF input)
- bands.in (NSCF bands input)
- bands_post.in (bands.x post-processing input)

Key settings (decided in STEP 4):
- ecutwfc = 120.0 Ry (60 Ha * 2, from Cd pseudopotential recommendation)
- ecutrho = 480.0 Ry (4 * ecutwfc)
- nbnd = 30 (> 22 occupied bands for 44 electrons, spin-unpolarized)
- occupations = 'fixed' (semiconductor, band gap ~0.59 eV)
- conv_thr = 1.0d-8 (Ry)
- K-mesh: scf_kmesh(prim, metal=False) from chem_llm.kpath
- Pseudopotentials: Cd (z_valence=20), Te (z_valence=24), total electrons=44
"""

import os
import shutil
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# Pseudopotential files (relative to calculations/NNN/)
# Cd: z_valence=20, Te: z_valence=24, total electrons=44
# Occupied bands = 44/2 = 22 (spin-unpolarized)
# nbnd must be > 22, use 30

# Cutoffs (INVARIANT 5: convert from Ha to Ry, 1 Ha = 2 Ry)
# Cd recommended: 60 Ha = 120 Ry
# Te recommended: 51 Ha = 102 Ry
# Use max: ecutwfc = 120 Ry
ECUTWFC_RY = 120.0
ECUTRHO_RY = 4 * ECUTWFC_RY  # 480 Ry

# Number of bands for bands calculation
# Must be > occupied bands (22), use 30
NBND = 30

# SCF convergence threshold (Ry)
CONV_THR = 1.0e-8

# Prefix for QE output files
PREFIX = 'cdte'

# Outdir for QE temporary files (relative to calculations/NNN/)
OUTDIR = './tmp'

# Pseudopotential directory (relative to calculations/NNN/)
# Pseudopotentials will be copied into calculations/NNN/
PSEUDO_DIR = '.'

# Pseudopotential filenames
PSEUDO_FILES = {
    'Cd': 'Cd.pbesol-srn-rrkjus_psl.1.0.0.UPF',
    'Te': 'Te.pbesol-srn-rrkjus_psl.1.0.0.UPF',
}

# Template directory where pseudopotentials are stored
TEMPLATE_DIR = 'template'


def setup_structure(nnn: str):
    """Set up QE input files for one structure."""
    calc_dir = f'calculations/{nnn}'
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(f'{calc_dir}/{OUTDIR}', exist_ok=True)
    
    # Copy pseudopotential files into calc_dir
    for elem, pp_file in PSEUDO_FILES.items():
        src = os.path.join(TEMPLATE_DIR, pp_file)
        dst = os.path.join(calc_dir, pp_file)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    # Read the structure
    cif_path = f'structures/structure_{nnn}.cif'
    atoms = ase.io.read(cif_path)
    
    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Structure {nnn}: expected 2 atoms, got {nat}"
    assert ntyp == 2, f"Structure {nnn}: expected 2 species, got {ntyp}"
    
    # Get the standard k-path and primitive cell (INVARIANT 3)
    # kp_atoms(kp) IS the primitive cell the k-path refers to
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)
    
    # Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))
    
    # SCF k-point mesh (from chem_llm.kpath, not hand-typed)
    kpts_scf = scf_kmesh(prim, metal=False)
    
    # Bands k-points (from the k-path)
    kpts_bands = kp_kpts(kp)
    
    # Pseudopotential mapping for write_espresso_in
    pseudopotentials = {
        'Cd': PSEUDO_FILES['Cd'],
        'Te': PSEUDO_FILES['Te'],
    }
    
    # SCF input (pw.in)
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
    
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=scf_input_data,
        pseudopotentials=pseudopotentials,
        kpts=kpts_scf,
    )
    
    # Bands input (bands.in) - NSCF calculation
    bands_input_data = {
        'calculation': 'bands',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': PSEUDO_DIR,
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
        'nbnd': NBND,
    }
    
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=bands_input_data,
        pseudopotentials=pseudopotentials,
        kpts=kpts_bands,
    )
    
    # Bands post-processing input (bands_post.in) for bands.x
    # &BANDS namelist: prefix, outdir, filband, lsym
    bands_post_content = f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{nnn}.bands.dat'
    lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)
    
    print(f"Setup complete for structure {nnn}")
    print(f"  K-path: {kp['path']}")
    print(f"  SCF k-mesh: {kpts_scf}")
    print(f"  Bands k-points: {len(kpts_bands)}")


if __name__ == '__main__':
    # Set up all 51 structures
    for i in range(51):
        nnn = f'{i:03d}'
        setup_structure(nnn)
    
    print("\nAll structures set up.")
    print("Next: run run_espresso_workflow on calculations/000")
