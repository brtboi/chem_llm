import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# =============================================================================
# setup_jobs.py
# For each structure NNN (000-050), create calculations/NNN/ containing:
#   - pseudopotential files (copied from template/)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input for pw.x)
#   - bands.in (NSCF bands input for pw.x)
#   - bands_post.in (bands.x post-processing input)
# =============================================================================

# --- Pseudopotential info (from get_pseudopotential results) -----------------
# B: z_valence=3.0, recommended_ecutwfc=38.0 Ha
# N: z_valence=5.0, recommended_ecutwfc=42.0 Ha
# INVARIANT 5: 1 Ha = 2 Ry
# ecutwfc = max(38.0, 42.0) * 2.0 = 84.0 Ry
# ecutrho = 4 * ecutwfc = 336.0 Ry

ECUTWFC_RY = 84.0
ECUTRHO_RY = 336.0

# --- Electron count and nbnd --------------------------------------------------
# 2 B atoms * 3 valence + 2 N atoms * 5 valence = 16 electrons
# Spin-unpolarized: occupied bands = 16/2 = 8
# nbnd must be >= 8; use 20 to have room for conduction bands
NBND = 20

# --- Common QE settings -------------------------------------------------------
PREFIX = 'bn'
OUTDIR = 'tmp'  # relative to calculations/NNN
PSEUDO_DIR = '.'  # pseudopotentials copied into calculations/NNN
CONV_THR = 1e-8

# --- Pseudopotential filenames ------------------------------------------------
PSEUDO_FILES = {
    'B': 'B.pbesol-rrkjus_psl.1.0.0.UPF',
    'N': 'N.pbesol-rrkjus_psl.1.0.0.UPF',
}

# --- Template directory -------------------------------------------------------
TEMPLATE_DIR = 'template'

# --- Structure directory ------------------------------------------------------
STRUCTURES_DIR = 'structures'

# --- Calculations directory ---------------------------------------------------
CALC_DIR = 'calculations'

# --- Number of structures -----------------------------------------------------
N_STRUCTURES = 51


def setup_one_structure(nnn: str):
    """Set up calculations/NNN/ for structure NNN."""
    calc_path = os.path.join(CALC_DIR, nnn)
    os.makedirs(calc_path, exist_ok=True)

    # Copy pseudopotential files into calculations/NNN/
    for elem, fname in PSEUDO_FILES.items():
        src = os.path.join(TEMPLATE_DIR, fname)
        dst = os.path.join(calc_path, fname)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Read the structure
    cif_path = os.path.join(STRUCTURES_DIR, f'structure_{nnn}.cif')
    atoms = ase.io.read(cif_path)

    # INVARIANT 1: nat and ntyp from the actual structure
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 4, f"Structure {nnn}: expected nat=4, got {nat}"
    assert ntyp == 2, f"Structure {nnn}: expected ntyp=2, got {ntyp}"

    # Get the standard k-path for this structure
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in

    # Save kpath.json
    kp_save(kp, os.path.join(calc_path, 'kpath.json'))

    # SCF k-point mesh (Monkhorst-Pack, insulator so metal=False)
    kpts_scf = scf_kmesh(prim, metal=False)

    # Bands k-points from the k-path
    kpts_bands = kp_kpts(kp)

    # --- Write pw.in (SCF) ----------------------------------------------------
    input_data_scf = {
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
        os.path.join(calc_path, 'pw.in'),
        prim,
        input_data=input_data_scf,
        pseudopotentials=PSEUDO_FILES,
        kpts=kpts_scf,
    )

    # --- Write bands.in (NSCF bands) ------------------------------------------
    input_data_bands = {
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
        os.path.join(calc_path, 'bands.in'),
        prim,
        input_data=input_data_bands,
        pseudopotentials=PSEUDO_FILES,
        kpts=kpts_bands,
    )

    # --- Write bands_post.in (bands.x) ----------------------------------------
    # bands.x input: &BANDS namelist with prefix, outdir, filband, lsym
    bands_post_content = f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{nnn}.bands.dat'
    lsym = .false.
/
"""
    with open(os.path.join(calc_path, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)

    print(f"Setup complete for calculations/{nnn}/")


def main():
    for i in range(N_STRUCTURES):
        nnn = f"{i:03d}"
        setup_one_structure(nnn)

    print(f"\nDone. Set up {N_STRUCTURES} calculation directories.")


if __name__ == '__main__':
    main()
