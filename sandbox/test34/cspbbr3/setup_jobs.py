#!/usr/bin/env python3
"""
setup_jobs.py -- Generate QE input files for all 51 CsPbBr3 structures.

For each structure NNN (000-050), creates calculations/NNN/ containing:
  - Pseudopotential files (copied from template/)
  - kpath.json (from chem_llm.kpath)
  - pw.in (SCF input, written with ase.io.espresso.write_espresso_in)
  - bands.in (NSCF bands input, written with write_espresso_in)
  - bands_post.in (bands.x post-processing input)

Key settings (see STEP 4 note):
  - ecutwfc = 88 Ry (max of 25/44/38 Ha * 2, INVARIANT 5)
  - ecutrho = 352 Ry (4 * ecutwfc)
  - noncolin=.true., lspinorb=.true. (spin-orbit, all PPs are fr)
  - occupations='fixed' (semiconductor, gap 1.78 eV)
  - conv_thr=1e-8
  - nbnd=100 (82 electrons with spin-orbit, each band holds 1 electron)
  - SCF k-mesh: scf_kmesh(prim, metal=False)
  - Bands k-points: kp_kpts(kp) from kpath

No pymatgen. ASE + chem_llm only.
"""

import os
import shutil
import numpy as np
from ase.io import read, write
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# --- Constants ---
N_STRUCTURES = 51
PREFIX = 'cspbbr3'
OUTDIR = 'tmp'

# Pseudopotential files (in template/)
PP_FILES = {
    'Cs': 'template/Cs.pbesol-rrkjus_psl.1.0.0.UPF',
    'Pb': 'template/Pb.pbesol-rrkjus_psl.1.0.0.UPF',
    'Br': 'template/Br.pbesol-rrkjus_psl.1.0.0.UPF',
}

# --- Read z_valence from each PP file (INVARIANT 2) ---
z_valence = {}
for elem, pp_path in PP_FILES.items():
    header = read_upf_header(pp_path)
    z_valence[elem] = int(header['z_valence'])
    print(f"{elem}: z_valence = {z_valence[elem]}")

# Total electrons (sum over all atoms in the primitive cell)
# We'll compute this per structure below.

# --- Cutoffs (INVARIANT 5: Ha -> Ry, 1 Ha = 2 Ry) ---
# Recommended cutoffs from get_pseudopotential (in Ha):
#   Cs: 25 Ha, Pb: 44 Ha, Br: 38 Ha
# Max = 44 Ha = 88 Ry
ECUTWFC_HA = 44.0  # max of the three recommended cutoffs
ECUTWFC_RY = 2.0 * ECUTWFC_HA  # 88 Ry
ECUTRHO_RY = 4.0 * ECUTWFC_RY  # 352 Ry
assert ECUTWFC_RY == 88.0, f"Expected 88 Ry, got {ECUTWFC_RY}"
assert ECUTRHO_RY == 352.0, f"Expected 352 Ry, got {ECUTRHO_RY}"

# --- nbnd ---
# With spin-orbit (noncolin), each band holds 1 electron.
# Total electrons = z_valence[Cs] + z_valence[Pb] + 3*z_valence[Br]
# = 9 + 22 + 3*17 = 82
# nbnd must be >= 82; use 100 for safety.
NBND = 100

# --- Common input_data for SCF and bands ---
# INVARIANT 6: pass a FLAT dict to write_espresso_in; ASE routes each key
# to the correct namelist (conv_thr -> &ELECTRONS, ecutwfc -> &SYSTEM, etc.)

def make_scf_input_data(kpts):
    """Build the flat input_data dict for the SCF calculation."""
    return {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': 1e-8,
        'noncolin': True,
        'lspinorb': True,
        'nspin': 1,
    }

def make_bands_input_data(kpts):
    """Build the flat input_data dict for the bands calculation."""
    return {
        'calculation': 'bands',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': 1e-8,
        'noncolin': True,
        'lspinorb': True,
        'nspin': 1,
        'nbnd': NBND,
    }

def write_bands_post_in(calc_dir, nnn):
    """Write bands_post.in (bands.x input) for structure NNN."""
    filband = f'{nnn}.bands.dat'
    content = f"""&BANDS
  prefix = '{PREFIX}'
  outdir = '{OUTDIR}'
  filband = '{filband}'
  lsym = .false.
/
EOF
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(content)

# --- Main loop ---
for i in range(N_STRUCTURES):
    nnn = f'{i:03d}'
    cif_path = f'structures/structure_{nnn}.cif'
    calc_dir = f'calculations/{nnn}'
    os.makedirs(calc_dir, exist_ok=True)

    # Copy pseudopotential files into calc_dir
    for elem, pp_src in PP_FILES.items():
        pp_dst = os.path.join(calc_dir, os.path.basename(pp_src))
        if not os.path.exists(pp_dst):
            shutil.copy2(pp_src, pp_dst)

    # Read the structure
    atoms = read(cif_path)

    # INVARIANT 1: nat/ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 5, f"Structure {nnn}: expected 5 atoms, got {nat}"
    assert ntyp == 3, f"Structure {nnn}: expected 3 species, got {ntyp}"

    # INVARIANT 3: k-path from chem_llm.kpath for THIS structure
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in

    # Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))

    # Compute total electrons from the actual structure (INVARIANT 2)
    total_electrons = sum(z_valence[sym] for sym in atoms.get_chemical_symbols())
    assert total_electrons == 82, f"Structure {nnn}: expected 82 electrons, got {total_electrons}"

    # SCF k-mesh (RULE: scf_kmesh, metal=False for semiconductor)
    scf_kpts = scf_kmesh(prim, metal=False)

    # Bands k-points from the k-path
    bands_kpts = kp_kpts(kp)

    # --- Write pw.in (SCF) ---
    scf_input_data = make_scf_input_data(scf_kpts)
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=scf_input_data,
        pseudopotentials={elem: os.path.basename(pp) for elem, pp in PP_FILES.items()},
        kpts=scf_kpts,
    )

    # --- Write bands.in (NSCF bands) ---
    bands_input_data = make_bands_input_data(bands_kpts)
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=bands_input_data,
        pseudopotentials={elem: os.path.basename(pp) for elem, pp in PP_FILES.items()},
        kpts=bands_kpts,
    )

    # --- Write bands_post.in (bands.x) ---
    write_bands_post_in(calc_dir, nnn)

    print(f"Structure {nnn}: nat={nat}, ntyp={ntyp}, electrons={total_electrons}, "
          f"scf_kpts={scf_kpts}, bands_kpts={len(bands_kpts)} points")

print("\nDone. All 51 calculation directories created.")
