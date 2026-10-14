import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# =============================================================================
# setup_jobs.py
# For each structure NNN (000..050), create calculations/NNN/ containing:
#   - pseudopotential files (copied from template/)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input, written with write_espresso_in)
#   - bands.in (NSCF bands input, written with write_espresso_in)
#   - bands_post.in (bands.x post-processing input)
#
# Key settings (see STEP 4 note):
#   ecutwfc = 92 Ry  (2 * 46 Ha, max of Mg=42 Ha and O=46 Ha; INVARIANT 5)
#   ecutrho = 368 Ry (4 * ecutwfc)
#   occupations = 'fixed' (MgO is an insulator, band_gap ~ 4.4 eV)
#   conv_thr = 1e-8 Ry
#   nspin = 1 (non-magnetic)
#   nbnd = 16 (2x occupied bands = 8; total valence electrons = 16)
#   SCF k-mesh: scf_kmesh(prim, metal=False)
#   Bands kpts: kp_kpts(kp)
# =============================================================================

# Pseudopotential files (relative to calculations/NNN/)
PP_FILES = {
    'Mg': 'Mg.pbesol-sr-pp.upf',
    'O': 'O.pbesol-sr-pp.upf',
}

# Read z_valence from the actual UPF files (INVARIANT 2)
# Mg: z_valence=10, O: z_valence=6 -> total = 16 electrons, 8 occupied bands
z_valences = {}
for elem, pp_file in PP_FILES.items():
    pp_path = os.path.join('template', pp_file)
    header = read_upf_header(pp_path)
    z_valences[elem] = header['z_valence']
    print(f"{elem}: z_valence = {z_valences[elem]}")

total_electrons = sum(z_valences.values())
print(f"Total valence electrons: {total_electrons}")

# INVARIANT 2: occupied bands = total_electrons / 2 for spin-unpolarized
n_occupied = int(total_electrons / 2)
print(f"Occupied bands (spin-unpolarized): {n_occupied}")

# nbnd: pad above occupied bands; use 2x occupied = 16
nbnd = 2 * n_occupied
assert nbnd >= n_occupied, f"nbnd={nbnd} must be >= n_occupied={n_occupied}"
print(f"nbnd = {nbnd}")

# ecutwfc: max of Mg (42 Ha) and O (46 Ha) = 46 Ha = 92 Ry (INVARIANT 5: 1 Ha = 2 Ry)
# Read from the UPF files to be safe
ecutwfc_ha = max(42.0, 46.0)  # from get_pseudopotential results
ecutwfc_ry = 2.0 * ecutwfc_ha
ecutrho_ry = 4.0 * ecutwfc_ry
print(f"ecutwfc = {ecutwfc_ry} Ry ({ecutwfc_ha} Ha)")
print(f"ecutrho = {ecutrho_ry} Ry")

N_STRUCTURES = 51  # 000..050

for i in range(N_STRUCTURES):
    nnn = f"{i:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)

    # Copy pseudopotential files into calculations/NNN/
    for elem, pp_file in PP_FILES.items():
        src = os.path.join('template', pp_file)
        dst = os.path.join(calc_dir, pp_file)
        shutil.copy2(src, dst)

    # Read the structure
    cif_path = f"structures/structure_{nnn}.cif"
    atoms = ase.io.read(cif_path)

    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Structure {nnn}: expected nat=2, got {nat}"
    assert ntyp == 2, f"Structure {nnn}: expected ntyp=2, got {ntyp}"

    # Get the standard k-path for THIS structure (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in

    # Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))

    # SCF k-mesh (RULE: never typed by hand)
    scf_kpts = scf_kmesh(prim, metal=False)
    print(f"Structure {nnn}: SCF k-mesh = {scf_kpts}")

    # Bands k-points from the k-path (INVARIANT 3)
    bands_kpts = kp_kpts(kp)

    # Pseudopotential filenames for write_espresso_in
    pseudopotentials = {elem: pp_file for elem, pp_file in PP_FILES.items()}

    # =====================================================================
    # pw.in (SCF)
    # =====================================================================
    scf_input_data = {
        'calculation': 'scf',
        'prefix': f'mgo_{nnn}',
        'outdir': './tmp',
        'pseudo_dir': '.',
        'ecutwfc': ecutwfc_ry,   # Ry
        'ecutrho': ecutrho_ry,   # Ry
        'occupations': 'fixed',
        'conv_thr': 1e-8,        # Ry
        'nspin': 1,
    }

    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=scf_input_data,
        pseudopotentials=pseudopotentials,
        kpts=scf_kpts,
    )

    # =====================================================================
    # bands.in (NSCF bands)
    # =====================================================================
    bands_input_data = {
        'calculation': 'bands',
        'prefix': f'mgo_{nnn}',
        'outdir': './tmp',
        'pseudo_dir': '.',
        'ecutwfc': ecutwfc_ry,   # Ry
        'ecutrho': ecutrho_ry,   # Ry
        'occupations': 'fixed',
        'conv_thr': 1e-8,        # Ry
        'nspin': 1,
        'nbnd': nbnd,
    }

    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=bands_input_data,
        pseudopotentials=pseudopotentials,
        kpts=bands_kpts,
    )

    # =====================================================================
    # bands_post.in (bands.x post-processing)
    # =====================================================================
    bands_post_content = f"""&BANDS
    prefix = 'mgo_{nnn}'
    outdir = './tmp'
    filband = '{nnn}.bands.dat'
    lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)

    print(f"Structure {nnn}: wrote pw.in, bands.in, bands_post.in, kpath.json")

print(f"\nDone. Set up {N_STRUCTURES} calculation directories.")
