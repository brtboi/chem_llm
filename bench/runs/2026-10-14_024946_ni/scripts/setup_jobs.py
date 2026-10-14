import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# =============================================================================
# setup_jobs.py
# For each structure NNN (000..050), create calculations/NNN/ containing:
#   - pseudopotential files (copied from template/)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input, written with write_espresso_in)
#   - bands.in (non-self-consistent bands input, written with write_espresso_in)
#   - bands_post.in (bands.x post-processing input)
# =============================================================================

# --- Pseudopotential settings ---
# Ni: z_valence = 18.0 (includes 3d semicore states)
# recommended_ecutwfc = 52.0 Ha (from get_pseudopotential, hint_level='normal')
# INVARIANT 5: 1 Ha = 2 Ry, so ecutwfc_ry = 2.0 * 52.0 = 104.0 Ry
# ecutrho = 4 * ecutwfc = 416.0 Ry (standard 4x ratio for local pseudopotentials)
ECUTWFC_RY = 104.0
ECUTRHO_RY = 416.0

# --- Electron count and nbnd ---
# 1 atom * 18.0 z_valence = 18 electrons total
# For spin-unpolarized (nspin=1), occupied bands = 18/2 = 9
# nbnd for bands: pad above 9, use nbnd=20 to be safe
NBND = 20

# --- Pseudopotential filename ---
# From get_pseudopotential: template/Ni.pbesol-sr-rrkjus_psl.1.0.0.UPF
PSEUDO_FILE = 'Ni.pbesol-sr-rrkjus_psl.1.0.0.UPF'
PSEUDO_SRC = os.path.join('template', PSEUDO_FILE)

# --- Number of structures ---
N_STRUCTURES = 51

# --- SCF k-point mesh ---
# Ni is a metal -> use scf_kmesh(prim, metal=True) for a denser mesh
# (RULE: k-path source)

# --- Common input_data for SCF and bands ---
# All variables are in a FLAT dict; write_espresso_in routes them to the
# correct namelist (INVARIANT 6).
# Units: ecutwfc/ecutrho in Ry, degauss in Ry, conv_thr dimensionless (Ry)

def make_scf_input_data(prefix):
    """Build the flat input_data dict for an SCF calculation."""
    return {
        'calculation': 'scf',
        'prefix': prefix,
        'outdir': 'tmp',
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'smearing',
        'smearing': 'mv',
        'degauss': 0.01,
        'conv_thr': 1.0e-8,
    }

def make_bands_input_data(prefix):
    """Build the flat input_data dict for a bands calculation."""
    return {
        'calculation': 'bands',
        'prefix': prefix,
        'outdir': 'tmp',
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'smearing',
        'smearing': 'mv',
        'degauss': 0.01,
        'conv_thr': 1.0e-8,
        'nbnd': NBND,
    }

def write_bands_post_in(path, prefix, nnn):
    """Write the bands.x post-processing input file.
    
    &BANDS namelist:
      prefix: must match the prefix used in pw.x
      outdir: must match the outdir used in pw.x
      filband: output file for band energies (NNN.bands.dat)
      lsym: .false. (skip band-representation file; not needed here)
    """
    with open(path, 'w') as f:
        f.write('&BANDS\n')
        f.write(f"  prefix = '{prefix}'\n")
        f.write(f"  outdir = 'tmp'\n")
        f.write(f"  filband = '{nnn}.bands.dat'\n")
        f.write('  lsym = .false.\n')
        f.write('/\n')
        f.write('\n')

for i in range(N_STRUCTURES):
    nnn = f'{i:03d}'
    calc_dir = f'calculations/{nnn}'
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotential into the calculation directory
    shutil.copy2(PSEUDO_SRC, os.path.join(calc_dir, PSEUDO_FILE))
    
    # Read the structure
    atoms = ase.io.read(f'structures/structure_{nnn}.cif')
    
    # INVARIANT 1: nat and ntyp computed from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    
    # Get the standard k-path for this structure
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in (INVARIANT 3)
    
    # Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))
    
    # SCF k-point mesh (metal=True for Ni)
    kpts_scf = scf_kmesh(prim, metal=True)
    
    # Bands k-points from the k-path (INVARIANT 3)
    kpts_bands = kp_kpts(kp)
    
    prefix = f'ni{nnn}'
    
    # Write pw.in (SCF)
    pw_in_path = os.path.join(calc_dir, 'pw.in')
    with open(pw_in_path, 'w') as f:
        write_espresso_in(
            f,
            prim,
            input_data=make_scf_input_data(prefix),
            pseudopotentials={'Ni': PSEUDO_FILE},
            kpts=kpts_scf,
        )
    
    # Write bands.in (non-self-consistent bands)
    bands_in_path = os.path.join(calc_dir, 'bands.in')
    with open(bands_in_path, 'w') as f:
        write_espresso_in(
            f,
            prim,
            input_data=make_bands_input_data(prefix),
            pseudopotentials={'Ni': PSEUDO_FILE},
            kpts=kpts_bands,
        )
    
    # Write bands_post.in (bands.x input)
    bands_post_path = os.path.join(calc_dir, 'bands_post.in')
    write_bands_post_in(bands_post_path, prefix, nnn)
    
    print(f'Wrote {calc_dir}/: pw.in, bands.in, bands_post.in, kpath.json, {PSEUDO_FILE}')

print(f'Done: {N_STRUCTURES} calculation directories created in calculations/')
