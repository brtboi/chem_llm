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
#   - Pseudopotential files (copied from template/)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input, written with write_espresso_in)
#   - bands.in (NSCF bands input, written with write_espresso_in)
#   - bands_post.in (bands.x post-processing input)
#
# Key settings (from STEP 4 note):
#   - ecutwfc = 114.0 Ry (57.0 Ha * 2, INVARIANT 5)
#   - ecutrho = 456.0 Ry (4 * ecutwfc)
#   - occupations = 'fixed' (GaN is a semiconductor)
#   - conv_thr = 1e-8 Ry
#   - nspin = 1 (non-magnetic)
#   - nbnd = 40 (well above 26 occupied bands)
#   - SCF k-mesh: computed via scf_kmesh(prim, metal=False)
#   - Bands k-path: kp_kpts(kp) from chem_llm.kpath
# =============================================================================

# Pseudopotential files
PSEUDO_FILES = {
    'Ga': 'template/Ga.pbesol-sr-rrkjus_psl.1.0.0.UPF',
    'N': 'template/N.pbesol-sr-rrkjus_psl.1.0.0.UPF',
}

# Read z_valence from the actual pseudopotential files (INVARIANT 2)
# This is the ground truth for the electron count, not a guess.
z_valence = {}
for elem, path in PSEUDO_FILES.items():
    header = read_upf_header(path)
    z_valence[elem] = header['z_valence']
    print(f"{elem}: z_valence = {z_valence[elem]}")

# ecutwfc: take the max recommended cutoff across all elements, convert Ha -> Ry
# Ga: 57.0 Ha, N: 42.0 Ha -> max = 57.0 Ha -> 114.0 Ry
ecutwfc_ha = max(57.0, 42.0)  # from get_pseudopotential results
ecutwfc_ry = 2.0 * ecutwfc_ha  # INVARIANT 5: 1 Ha = 2 Ry
ecutrho_ry = 4.0 * ecutwfc_ry
print(f"ecutwfc = {ecutwfc_ry} Ry ({ecutwfc_ha} Ha)")
print(f"ecutrho = {ecutrho_ry} Ry")

# nbnd: total electrons / 2 for spin-unpolarized run
# Total electrons = 2 * z_valence[Ga] + 2 * z_valence[N] = 2*21 + 2*5 = 52
# Occupied bands = 52/2 = 26
# Set nbnd = 40 (well above 26, room for conduction bands)
nbnd = 40
print(f"nbnd = {nbnd} (occupied bands = 26)")

N_STRUCTURES = 51  # 000-050

for i in range(N_STRUCTURES):
    nnn = f"{i:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotential files into the calculation directory
    for elem, src in PSEUDO_FILES.items():
        dst = os.path.join(calc_dir, os.path.basename(src))
        shutil.copy2(src, dst)
    
    # Read the structure
    atoms = ase.io.read(f"structures/structure_{nnn}.cif")
    
    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 4, f"Structure {nnn}: expected nat=4, got {nat}"
    assert ntyp == 2, f"Structure {nnn}: expected ntyp=2, got {ntyp}"
    
    # Get the k-path and primitive cell (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    
    # Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))
    
    # SCF k-mesh (computed, not hand-typed)
    kpts_scf = scf_kmesh(prim, metal=False)
    print(f"Structure {nnn}: SCF k-mesh = {kpts_scf}")
    
    # Pseudopotential filenames (relative to calc_dir, since pseudo_dir='.')
    pseudo_dict = {
        'Ga': os.path.basename(PSEUDO_FILES['Ga']),
        'N': os.path.basename(PSEUDO_FILES['N']),
    }
    
    # SCF input (pw.in)
    scf_input_data = {
        'calculation': 'scf',
        'prefix': 'pwscf',
        'outdir': 'tmp',
        'pseudo_dir': '.',
        'ecutwfc': ecutwfc_ry,
        'ecutrho': ecutrho_ry,
        'occupations': 'fixed',
        'conv_thr': 1e-8,
        'nspin': 1,
    }
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=scf_input_data,
        pseudopotentials=pseudo_dict,
        kpts=kpts_scf,
    )
    
    # Bands input (bands.in)
    bands_input_data = {
        'calculation': 'bands',
        'prefix': 'pwscf',
        'outdir': 'tmp',
        'pseudo_dir': '.',
        'ecutwfc': ecutwfc_ry,
        'ecutrho': ecutrho_ry,
        'occupations': 'fixed',
        'nspin': 1,
        'nbnd': nbnd,
    }
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=bands_input_data,
        pseudopotentials=pseudo_dict,
        kpts=kp_kpts(kp),
    )
    
    # Bands post-processing input (bands_post.in)
    # bands.x input: &BANDS namelist with prefix, outdir, filband, lsym
    bands_post_content = f"""&BANDS
    prefix = 'pwscf'
    outdir = 'tmp'
    filband = '{nnn}.bands.dat'
    lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)
    
    if i < 3 or i == N_STRUCTURES - 1:
        print(f"Wrote {calc_dir}/: pw.in, bands.in, bands_post.in, kpath.json, pseudos")

print(f"\nSet up {N_STRUCTURES} calculation directories: calculations/000 to calculations/{N_STRUCTURES-1:03d}")
print("Done.")
