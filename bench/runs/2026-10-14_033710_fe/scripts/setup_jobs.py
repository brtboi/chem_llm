import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# =============================================================================
# setup_jobs.py
# Creates calculations/NNN/ directories for each structure with:
#   - pseudopotential files (copied from template/)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input)
#   - bands.in (bands input)
#   - bands_post.in (bands.x post-processing input)
# =============================================================================

# --- Pseudopotential info ---
pp_file = 'template/Fe.pbesol-sr-pp.upf'
pp_header = read_upf_header(pp_file)
z_valence = pp_header['z_valence']  # 16.0 for Fe (includes semicore 3p)
print(f"Pseudopotential z_valence: {z_valence}")

# --- Cutoff conversion (INVARIANT 5) ---
# get_pseudopotential returns recommended_ecutwfc in HARTREE
# QE's ecutwfc is in RYDBERG: 1 Ha = 2 Ry
ecutwfc_ha = 59.0  # from get_pseudopotential result
ecutwfc_ry = 2.0 * ecutwfc_ha  # 118.0 Ry
ecutrho_ry = 4.0 * ecutwfc_ry  # 472.0 Ry
print(f"ecutwfc: {ecutwfc_ha} Ha = {ecutwfc_ry} Ry")
print(f"ecutrho: {ecutrho_ry} Ry")

# --- nbnd calculation ---
# For spin-polarized (nspin=2): each spin channel has z_valence/2 electrons
# Fe: 16 electrons total -> 8 per spin channel -> 4 occupied bands per spin
# nbnd should be well above this. Set nbnd=32 (8x margin)
n_electrons_total = int(z_valence)  # 16 for 1 Fe atom
n_electrons_per_spin = n_electrons_total // 2  # 8
n_occupied_per_spin = n_electrons_per_spin // 2  # 4
nbnd = 32
assert nbnd > n_occupied_per_spin, f"nbnd={nbnd} must be > n_occupied_per_spin={n_occupied_per_spin}"
print(f"nbnd: {nbnd} (n_occupied_per_spin={n_occupied_per_spin})")

# --- SCF k-mesh ---
# For a metal (FM Fe), use metal=True for ~2x density
# This will be computed per structure in the loop

# --- Input data for SCF ---
scf_input_data = {
    'calculation': 'scf',
    'prefix': 'fe',
    'outdir': './tmp',
    'pseudo_dir': '../../template',
    'ecutwfc': ecutwfc_ry,
    'ecutrho': ecutrho_ry,
    'occupations': 'smearing',
    'smearing': 'mv',
    'degauss': 0.01,
    'conv_thr': 1.0e-8,
    'nspin': 2,
}

# --- Input data for bands ---
bands_input_data = {
    'calculation': 'bands',
    'prefix': 'fe',
    'outdir': './tmp',
    'pseudo_dir': '../../template',
    'ecutwfc': ecutwfc_ry,
    'ecutrho': ecutrho_ry,
    'nspin': 2,
    'nbnd': nbnd,
}

# --- Bands post-processing input ---
# Written manually as a simple namelist file

def write_bands_post_in(path, nnn):
    """Write bands.x post-processing input file."""
    content = f"""&BANDS
    prefix = 'fe',
    outdir = './tmp',
    filband = '{nnn}.bands.dat',
    lsym = .false.,
    spin_component = 1
/
"""
    with open(path, 'w') as f:
        f.write(content)

# --- Main loop over structures ---
for nnn in range(51):
    nnn_str = f"{nnn:03d}"
    calc_dir = f"calculations/{nnn_str}"
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(f"{calc_dir}/tmp", exist_ok=True)
    
    # Copy pseudopotential files
    shutil.copy(pp_file, f"{calc_dir}/Fe.pbesol-sr-pp.upf")
    
    # Read the structure
    cif_file = f"structures/structure_{nnn_str}.cif"
    atoms = ase.io.read(cif_file)
    
    # Get the k-path and primitive cell (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)
    
    # Save kpath.json
    kp_save(kp, f"{calc_dir}/kpath.json")
    
    # Set initial magnetic moments (magnetism rule: FM Fe, moment 2.645 muB)
    # 0.5 is a fraction of valence electrons (QE units)
    prim.set_initial_magnetic_moments([0.5 if s == 'Fe' else 0.0 for s in prim.get_chemical_symbols()])
    
    # Compute SCF k-mesh (metal=True for FM Fe)
    kpts_scf = scf_kmesh(prim, metal=True)
    
    # Write pw.in (SCF)
    write_espresso_in(
        f"{calc_dir}/pw.in",
        prim,
        input_data=scf_input_data,
        pseudopotentials={'Fe': 'Fe.pbesol-sr-pp.upf'},
        kpts=kpts_scf,
    )
    
    # Write bands.in
    kpts_bands = kp_kpts(kp)
    write_espresso_in(
        f"{calc_dir}/bands.in",
        prim,
        input_data=bands_input_data,
        pseudopotentials={'Fe': 'Fe.pbesol-sr-pp.upf'},
        kpts=kpts_bands,
    )
    
    # Write bands_post.in
    write_bands_post_in(f"{calc_dir}/bands_post.in", nnn_str)
    
    print(f"Wrote {calc_dir}/ (pw.in, bands.in, bands_post.in, kpath.json, Fe.pbesol-sr-pp.upf)")

print("\nDone setting up all 51 calculation directories.")
