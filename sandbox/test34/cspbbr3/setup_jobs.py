"""
setup_jobs.py -- Generate QE input files for all 51 CsPbBr3 structures.

For each structure NNN (000..050):
  calculations/NNN/
    Cs.pbesol-rrkjus_psl.1.0.0.UPF
    Pb.pbesol-rrkjus_psl.1.0.0.UPF
    Br.pbesol-rrkjus_psl.1.0.0.UPF
    kpath.json
    pw.in          (SCF)
    bands.in       (NSCF bands)
    bands_post.in  (bands.x post-processing)

Key decisions (from STEP 4 note):
  - Pseudopotentials: fully relativistic (fr), pbesol, nc, stringent, UPF
    Cs: z_valence=9, Pb: z_valence=22, Br: z_valence=17
    Total electrons per formula unit = 9 + 22 + 3*17 = 82
  - Spin-orbit: noncolin=.true., lspinorb=.true.
    With SO, each band holds 1 electron, so n_occupied = 82.
  - Cutoffs (INVARIANT 5: Ha -> Ry, 1 Ha = 2 Ry):
    Recommended cutoffs from get_pseudopotential (Pseudo-Dojo, hint_level='normal'):
      Cs: 25 Ha, Pb: 44 Ha, Br: 38 Ha
    Max = 44 Ha (Pb) -> ecutwfc = 88 Ry
    ecutrho = 4 * 88 = 352 Ry
  - occupations='fixed' (insulator, gap ~1.6-2 eV without SO, ~0.4-1.6 with SO)
  - conv_thr=1.0d-8 (in &ELECTRONS, routed by ASE)
  - SCF k-mesh: scf_kmesh(prim, metal=False) from chem_llm.kpath
  - Bands kpts: kp_kpts(kp) -- ndarray shape (n_kpts, 4) per ASE docs
  - nbnd=100 (pad above n_occupied=82)
  - bands_post.in: &BANDS with prefix, outdir, filband, lsym=.false.
"""

import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# --- Paths ---
STRUCTURES_DIR = 'structures'
TEMPLATE_DIR = 'template'
CALC_DIR = 'calculations'

# Pseudopotential files (copied into each calculations/NNN/)
PP_FILES = {
    'Cs': 'Cs.pbesol-rrkjus_psl.1.0.0.UPF',
    'Pb': 'Pb.pbesol-rrkjus_psl.1.0.0.UPF',
    'Br': 'Br.pbesol-rrkjus_psl.1.0.0.UPF',
}

# --- Read z_valence from the actual UPF files (INVARIANT 2) ---
# Sum z_valence over every atom to get the total electron count.
# With spin-orbit, each band holds 1 electron, so n_occupied = n_electrons.
z_valences = {}
for elem, fname in PP_FILES.items():
    pp_path = os.path.join(TEMPLATE_DIR, fname)
    header = read_upf_header(pp_path)
    z_valences[elem] = header['z_valence']
    print(f"{elem}: z_valence = {z_valences[elem]}")

# Total electrons per formula unit (CsPbBr3: 1 Cs + 1 Pb + 3 Br)
n_electrons = z_valences['Cs'] + z_valences['Pb'] + 3 * z_valences['Br']
print(f"Total electrons per formula unit = {n_electrons}")

# With spin-orbit (noncolin=.true., lspinorb=.true.), each band holds 1 electron
n_occupied = int(n_electrons)  # = 82
print(f"n_occupied (with SO) = {n_occupied}")

# nbnd must be >= n_occupied; pad above it
nbnd = 100
assert nbnd >= n_occupied, f"nbnd={nbnd} < n_occupied={n_occupied}"

# --- Cutoffs (INVARIANT 5: Ha -> Ry, 1 Ha = 2 Ry) ---
# Recommended cutoffs from get_pseudopotential (Pseudo-Dojo, hint_level='normal'):
# These are NOT stored in the UPF file itself -- they are Pseudo-Dojo's
# recommendation for this specific pseudopotential table.
# Cs: 25 Ha, Pb: 44 Ha, Br: 38 Ha
ecut_ha = {
    'Cs': 25.0,
    'Pb': 44.0,
    'Br': 38.0,
}

# Use the maximum recommended cutoff (in Ha), convert to Ry
max_ecut_ha = max(ecut_ha.values())
ecutwfc_ry = 2.0 * max_ecut_ha  # 1 Ha = 2 Ry
ecutrho_ry = 4.0 * ecutwfc_ry
print(f"Max ecut hint = {max_ecut_ha} Ha -> ecutwfc = {ecutwfc_ry} Ry, ecutrho = {ecutrho_ry} Ry")

# --- QE input parameters (flat dict, ASE routes to correct namelists) ---
# INVARIANT 6: pass a flat dict, let ASE route each key to its namelist
input_data_scf = {
    'calculation': 'scf',
    'prefix': 'cspbbr3',
    'outdir': 'tmp_cspbbr3',
    'pseudo_dir': '.',
    'ecutwfc': ecutwfc_ry,
    'ecutrho': ecutrho_ry,
    'occupations': 'fixed',
    'conv_thr': 1.0e-8,
    'noncolin': True,
    'lspinorb': True,
    'nspin': 1,
}

input_data_bands = {
    'calculation': 'bands',
    'prefix': 'cspbbr3',
    'outdir': 'tmp_cspbbr3',
    'pseudo_dir': '.',
    'ecutwfc': ecutwfc_ry,
    'ecutrho': ecutrho_ry,
    'occupations': 'fixed',
    'conv_thr': 1.0e-8,
    'noncolin': True,
    'lspinorb': True,
    'nspin': 1,
    'nbnd': nbnd,
}

# --- Process each structure ---
N_STRUCTURES = 51

for i in range(N_STRUCTURES):
    nnn = f"{i:03d}"
    calc_path = os.path.join(CALC_DIR, nnn)
    os.makedirs(calc_path, exist_ok=True)

    # Copy pseudopotential files into calculations/NNN/
    for elem, fname in PP_FILES.items():
        src = os.path.join(TEMPLATE_DIR, fname)
        dst = os.path.join(calc_path, fname)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Read the structure
    cif_path = os.path.join(STRUCTURES_DIR, f"structure_{nnn}.cif")
    atoms = ase.io.read(cif_path)

    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 5, f"Structure {nnn}: expected 5 atoms, got {nat}"
    assert ntyp == 3, f"Structure {nnn}: expected 3 species, got {ntyp}"

    # INVARIANT 3: k-path from chem_llm.kpath for THIS structure
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    kp_save(kp, os.path.join(calc_path, 'kpath.json'))

    # SCF k-mesh (Monkhorst-Pack, insulator -> metal=False)
    kpts_scf = scf_kmesh(prim, metal=False)
    print(f"Structure {nnn}: SCF k-mesh = {kpts_scf}")

    # Bands k-points from kpath (ndarray shape (n_kpts, 4))
    kpts_bands = kp_kpts(kp)
    print(f"Structure {nnn}: bands k-points = {kpts_bands.shape}")

    # Write pw.in (SCF)
    pw_in_path = os.path.join(calc_path, 'pw.in')
    with open(pw_in_path, 'w') as f:
        write_espresso_in(
            f, prim, input_data=input_data_scf,
            pseudopotentials=PP_FILES,
            kpts=kpts_scf,
        )

    # Write bands.in (NSCF bands)
    bands_in_path = os.path.join(calc_path, 'bands.in')
    with open(bands_in_path, 'w') as f:
        write_espresso_in(
            f, prim, input_data=input_data_bands,
            pseudopotentials=PP_FILES,
            kpts=kpts_bands,
        )

    # Write bands_post.in (bands.x input)
    # &BANDS namelist: prefix, outdir, filband, lsym=.false.
    bands_post_path = os.path.join(calc_path, 'bands_post.in')
    with open(bands_post_path, 'w') as f:
        f.write("&BANDS\n")
        f.write(f"  prefix = 'cspbbr3'\n")
        f.write(f"  outdir = 'tmp_cspbbr3'\n")
        f.write(f"  filband = '{nnn}.bands.dat'\n")
        f.write(f"  lsym = .false.\n")
        f.write("/\n")

    print(f"Structure {nnn}: wrote pw.in, bands.in, bands_post.in, kpath.json")

print(f"\nDone. Set up {N_STRUCTURES} calculation directories in {CALC_DIR}/")
