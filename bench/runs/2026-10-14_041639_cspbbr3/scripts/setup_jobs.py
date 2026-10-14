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
#   - pw.in (SCF input)
#   - bands.in (NSCF bands input)
#   - bands_post.in (bands.x post-processing input)
# =============================================================================

# --- Pseudopotential data (from STEP 3) ---
# z_valence: Cs=9, Pb=22, Br=17
# Total electrons = 9 + 22 + 3*17 = 82
# Occupied bands (spin-unpolarized) = 82/2 = 41
# nbnd for bands: 64 (comfortable margin above 41)

# --- Cutoff conversion (INVARIANT 5) ---
# Recommended ecutwfc (Ha): Cs=25, Pb=44, Br=38
# Max = 44.0 Ha (Pb)
# ecutwfc = 2.0 * 44.0 = 88.0 Ry
# ecutrho = 4 * 88.0 = 352.0 Ry
ECUTWFC_RY = 88.0
ECUTRHO_RY = 352.0

# --- SCF settings ---
PREFIX = 'cspbbr3'
OUTDIR = 'tmp_cspbbr3'
CONV_THR = 1.0e-8  # Ry
NBND = 64

# --- Pseudopotential filenames ---
PP_FILES = {
    'Cs': 'Cs.pbesol-sr-rrkjus_psl.1.0.0.UPF',
    'Pb': 'Pb.pbesol-sr-rrkjus_psl.1.0.0.UPF',
    'Br': 'Br.pbesol-sr-rrkjus_psl.1.0.0.UPF',
}

# --- Verify z_valence from actual UPF files (INVARIANT 2) ---
for sp, fname in PP_FILES.items():
    pp_path = os.path.join('template', fname)
    header = read_upf_header(pp_path)
    print(f'{sp}: z_valence={header["z_valence"]}, relativistic={header["relativistic"]}')

# --- Read base structure (structure_000.cif is the primitive cell) ---
base = ase.io.read('structures/structure_000.cif')
n_base = len(base)
ntyp_base = len(set(base.get_chemical_symbols()))
print(f'Base structure: {n_base} atoms, {ntyp_base} species')

# --- Process each structure ---
for i in range(51):
    nnn = f'{i:03d}'
    calc_dir = f'calculations/{nnn}'
    os.makedirs(calc_dir, exist_ok=True)

    # Copy pseudopotential files into calculations/NNN/
    for sp, fname in PP_FILES.items():
        src = os.path.join('template', fname)
        dst = os.path.join(calc_dir, fname)
        shutil.copy2(src, dst)

    # Read the structure
    atoms = ase.io.read(f'structures/structure_{nnn}.cif')

    # INVARIANT 1: nat and ntyp from the actual ASE Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == n_base, f'structure_{nnn}: nat {nat} != base {n_base}'
    assert ntyp == ntyp_base, f'structure_{nnn}: ntyp {ntyp} != base {ntyp_base}'

    # INVARIANT 3: k-path from chem_llm.kpath for THIS structure
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))

    # SCF k-mesh (RULE: never typed by hand)
    kpts_scf = scf_kmesh(prim, metal=False)
    print(f'structure_{nnn}: SCF k-mesh = {kpts_scf}')

    # --- Write pw.in (SCF) ---
    input_data_scf = {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
        'nspin': 1,
    }
    with open(os.path.join(calc_dir, 'pw.in'), 'w') as f:
        write_espresso_in(
            f, prim, input_data=input_data_scf,
            pseudopotentials=PP_FILES, kpts=kpts_scf
        )

    # --- Write bands.in (NSCF bands) ---
    # kpts must be a numpy array of shape (n_kpts, 4) with weights
    kpts_bands = kp_kpts(kp)
    # Ensure it's a numpy array of shape (n_kpts, 4)
    kpts_bands = np.array(kpts_bands)
    if kpts_bands.ndim == 2 and kpts_bands.shape[1] == 3:
        # Add weight column (all 1s for a band path)
        kpts_bands = np.hstack([kpts_bands, np.ones((kpts_bands.shape[0], 1))])
    assert kpts_bands.shape[1] == 4, f'kpts_bands shape {kpts_bands.shape} != (n, 4)'

    input_data_bands = {
        'calculation': 'bands',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
        'nspin': 1,
        'nbnd': NBND,
    }
    with open(os.path.join(calc_dir, 'bands.in'), 'w') as f:
        write_espresso_in(
            f, prim, input_data=input_data_bands,
            pseudopotentials=PP_FILES, kpts=kpts_bands
        )

    # --- Write bands_post.in (bands.x input) ---
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

    print(f'structure_{nnn}: wrote pw.in, bands.in, bands_post.in, kpath.json')

print('Done: all 51 calculation directories set up.')
