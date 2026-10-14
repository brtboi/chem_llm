"""
setup_jobs.py -- Generate QE input files for all 51 Bi2Se3 structures.

For each structure NNN (000-050):
  calculations/NNN/
    pw.in          -- SCF input (write_espresso_in)
    bands.in       -- NSCF bands input (write_espresso_in)
    bands_post.in  -- bands.x post-processing input
    kpath.json     -- k-path from chem_llm.kpath
    Bi.pbesol-rrkjus_psl.1.0.0.UPF  -- pseudopotential (copied from template/)
    Se.pbesol-rrkjus_psl.1.0.0.UPF  -- pseudopotential (copied from template/)

Key decisions (see STEP 4 note):
  - ecutwfc = 104.0 Ry (max of Bi 44 Ha=88 Ry, Se 52 Ha=104 Ry; INVARIANT 5)
  - ecutrho = 416.0 Ry (4 * ecutwfc)
  - occupations='smearing', smearing='mv', degauss=0.01 Ry (narrow-gap semiconductor)
  - noncolin=.true., lspinorb=.true. (spin-orbit, Bi Z=83>54)
  - nbnd=130 (n_electrons=118, SO: 1 e/band, so n_occupied=118, nbnd>=118)
  - SCF k-mesh: scf_kmesh(prim, metal=True) from chem_llm.kpath
  - bands.in kpts: kp_kpts(kp) from chem_llm.kpath (INVARIANT 3)
  - bands_post.in: lsym=.false. (skip band-representation file, saves ~20 min with SO)
"""

import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# Pseudopotential files (from template/)
PP_BI = 'Bi.pbesol-rrkjus_psl.1.0.0.UPF'
PP_SE = 'Se.pbesol-rrkjus_psl.1.0.0.UPF'

# Pseudopotential directory relative to calculations/NNN
# template/ is at the project root, calculations/NNN is two levels deep
PSEUDO_DIR = '../../template'

# QE settings (all units in Ry unless noted)
ECUTWFC = 104.0   # Ry (max of Bi 44 Ha=88 Ry, Se 52 Ha=104 Ry; INVARIANT 5: 1 Ha = 2 Ry)
ECUTRHO = 416.0   # Ry (4 * ecutwfc)
CONV_THR = 1e-8   # Ry (SCF convergence threshold)
DEGAUSS = 0.01    # Ry (smearing width, mv)
NBND = 130        # bands (n_electrons=118, SO: 1 e/band, pad above)

# Common input_data for SCF and bands (flat dict -- ASE routes to correct namelists, INVARIANT 6)
common_input_data = {
    'prefix': 'bi2se3',
    'outdir': './tmp',
    'pseudo_dir': PSEUDO_DIR,
    'ecutwfc': ECUTWFC,
    'ecutrho': ECUTRHO,
    'occupations': 'smearing',
    'smearing': 'mv',
    'degauss': DEGAUSS,
    'conv_thr': CONV_THR,
    'noncolin': True,
    'lspinorb': True,
}

# Pseudopotential mapping for write_espresso_in
pseudopotentials = {
    'Bi': PP_BI,
    'Se': PP_SE,
}

for i in range(51):
    nnn = f"{i:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)

    # Copy pseudopotential files into calc_dir (pw.x runs inside calc_dir)
    for pp in [PP_BI, PP_SE]:
        src = f"template/{pp}"
        dst = f"{calc_dir}/{pp}"
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Read the structure
    atoms = ase.io.read(f"structures/structure_{nnn}.cif")

    # INVARIANT 1: nat/ntyp from actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 5, f"{nnn}: expected 5 atoms, got {nat}"
    assert ntyp == 2, f"{nnn}: expected 2 species, got {ntyp}"

    # INVARIANT 3: k-path from chem_llm.kpath for THIS structure
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in

    # INVARIANT 1: nat/ntyp from the primitive cell (what we actually write)
    prim_nat = len(prim)
    prim_ntyp = len(set(prim.get_chemical_symbols()))
    assert prim_nat == 5, f"{nnn}: primitive expected 5 atoms, got {prim_nat}"
    assert prim_ntyp == 2, f"{nnn}: primitive expected 2 species, got {prim_ntyp}"

    # Save kpath.json (holds labels, ticks, segments -- no separate labels file)
    kp_save(kp, f"{calc_dir}/kpath.json")

    # SCF k-mesh: metal=True for narrow-gap semiconductor (denser mesh)
    kpts_scf = scf_kmesh(prim, metal=True)

    # --- pw.in (SCF) ---
    scf_input_data = dict(common_input_data)
    scf_input_data['calculation'] = 'scf'
    write_espresso_in(
        f"{calc_dir}/pw.in",
        prim,
        input_data=scf_input_data,
        pseudopotentials=pseudopotentials,
        kpts=kpts_scf,
    )

    # --- bands.in (NSCF bands) ---
    bands_input_data = dict(common_input_data)
    bands_input_data['calculation'] = 'bands'
    bands_input_data['nbnd'] = NBND
    # INVARIANT 3: kpts from kp_kpts(kp) -- the k-path k-points
    kpts_bands = kp_kpts(kp)
    write_espresso_in(
        f"{calc_dir}/bands.in",
        prim,
        input_data=bands_input_data,
        pseudopotentials=pseudopotentials,
        kpts=kpts_bands,
    )

    # --- bands_post.in (bands.x post-processing) ---
    # Written as a plain namelist file (bands.x has a different input format than pw.x)
    bands_post_content = f"""&BANDS
  prefix = '{common_input_data['prefix']}'
  outdir = '{common_input_data['outdir']}'
  filband = '{nnn}.bands.dat'
  lsym = .false.
  spin_component = 1
/
"""
    with open(f"{calc_dir}/bands_post.in", 'w') as f:
        f.write(bands_post_content)

    print(f"{nnn}: pw.in, bands.in, bands_post.in, kpath.json written to {calc_dir}/")

print("\nDone. All 51 calculation directories set up.")
