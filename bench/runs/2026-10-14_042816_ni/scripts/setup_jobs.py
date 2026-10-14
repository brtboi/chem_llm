import os
import shutil
import numpy as np
import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from ase.io.espresso import write_espresso_in

# =============================================================================
# setup_jobs.py
# For each structure NNN (000-050), create calculations/NNN/ containing:
#   - Ni.pbesol_srr-rrkjus_psl.1.0.0.UPF (copied from template/)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input)
#   - bands.in (NSCF bands input)
#   - bands_post.in (bands.x post-processing input)
# =============================================================================

# --- Pseudopotential ---
PP_FILE = 'template/Ni.pbesol_srr-rrkjus_psl.1.0.0.UPF'
PP_NAME = 'Ni.pbesol_srr-rrkjus_psl.1.0.0.UPF'

# --- Cutoffs (INVARIANT 5: convert from Ha to Ry) ---
# get_pseudopotential reported recommended_ecutwfc = 52.0 Ha
# 1 Ha = 2 Ry, so ecutwfc = 104.0 Ry
ECUTWFC_HA = 52.0
ECUTWFC_RY = 2.0 * ECUTWFC_HA  # 104.0 Ry
ECUTRHO_RY = 4.0 * ECUTWFC_RY  # 416.0 Ry (4x ecutwfc, standard for PAW/NC)
assert ECUTWFC_RY == 104.0, f"ecutwfc conversion error: {ECUTWFC_RY}"
assert ECUTRHO_RY == 416.0, f"ecutrho conversion error: {ECUTRHO_RY}"

# --- SCF settings ---
# Ni is a metal (band_gap_eV=0.0) and magnetic (0.707 muB/f.u.)
# -> spin-polarized (nspin=2) + smearing (mv, degauss=0.01 Ry)
# z_valence=18, 1 atom -> 18 electrons -> 9 occupied bands per spin channel
# nbnd=20 (well above 9)
N_SPIN = 2
NBND = 20
DEGAUSS = 0.01  # Ry
CONV_THR = 1e-8  # Ry

# --- Magnetic moment ---
# initial_magnetization is a fraction of valence electrons (QE units)
# 0.5 * 18 = 9 muB per atom, close to MP's 0.707 muB/f.u. for 1 atom
# Actually, 0.707 muB is the experimental moment. In QE units (fraction of valence),
# 0.707/18 = 0.0393. But the convention is to use a larger initial moment to help
# convergence. 0.5 is a reasonable starting value for a transition metal.
INITIAL_MOMENT = 0.5  # fraction of valence electrons

# --- Prefix and outdir ---
PREFIX = 'ni'
OUTDIR = './tmp'

# --- Pseudopotential directory ---
# pw.x runs inside calculations/NNN, so pseudo_dir='.' (PP copied there)
PSEUDO_DIR = '.'


def setup_structure(nnn: str):
    """Set up calculations/NNN/ for structure NNN."""
    calc_dir = f'calculations/{nnn}'
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(f'{calc_dir}/{OUTDIR}', exist_ok=True)

    # Copy pseudopotential
    shutil.copy(PP_FILE, f'{calc_dir}/{PP_NAME}')

    # Read structure
    atoms = ase.io.read(f'structures/structure_{nnn}.cif')
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    print(f"Structure {nnn}: nat={nat}, ntyp={ntyp}")

    # Get k-path and primitive cell (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)
    kp_save(kp, f'{calc_dir}/kpath.json')

    # Set initial magnetic moments on the primitive cell
    # (kp_atoms returns a cell with zero moments, so we must set them)
    prim.set_initial_magnetic_moments(
        [INITIAL_MOMENT if s == 'Ni' else 0.0 for s in prim.get_chemical_symbols()]
    )

    # --- SCF k-mesh (metal -> denser) ---
    kpts_scf = scf_kmesh(prim, metal=True)
    print(f"  SCF k-mesh: {kpts_scf}")

    # --- Bands k-points ---
    kpts_bands = kp_kpts(kp)
    print(f"  Bands k-points: {len(kpts_bands)}")

    # --- Pseudopotential mapping ---
    pseudos = {'Ni': PP_NAME}

    # --- SCF input (pw.in) ---
    input_data_scf = {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': PSEUDO_DIR,
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'nspin': N_SPIN,
        'occupations': 'smearing',
        'smearing': 'mv',
        'degauss': DEGAUSS,
        'conv_thr': CONV_THR,
    }
    write_espresso_in(
        f'{calc_dir}/pw.in',
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
        'pseudo_dir': PSEUDO_DIR,
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'nspin': N_SPIN,
        'nbnd': NBND,
    }
    write_espresso_in(
        f'{calc_dir}/bands.in',
        prim,
        input_data=input_data_bands,
        pseudopotentials=pseudos,
        kpts=kpts_bands,
    )

    # --- Bands post-processing input (bands_post.in) ---
    # Written manually as a simple namelist file
    bands_post_content = f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{nnn}.bands.dat'
    lsym = .false.
    spin_component = 1
/
"""
    with open(f'{calc_dir}/bands_post.in', 'w') as f:
        f.write(bands_post_content)

    print(f"  Wrote {calc_dir}/pw.in, bands.in, bands_post.in")


if __name__ == '__main__':
    for i in range(51):
        nnn = f'{i:03d}'
        setup_structure(nnn)
    print("\nDone. Set up 51 calculation directories.")
