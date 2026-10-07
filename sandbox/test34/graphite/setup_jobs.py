import os
import shutil
import numpy as np
from ase.io import read, write
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save

# =============================================================================
# setup_jobs.py
# For each structure NNN (000-050), create calculations/NNN/ containing:
#   - C.pbesol-sr-rrkjus_psl.1.upf (pseudopotential, copied from template/)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input, written with write_espresso_in)
#   - bands.in (bands input, written with write_espresso_in)
#   - bands_post.in (bands.x input, written with write_fortran_namelist)
# =============================================================================

# --- Pseudopotential info (from get_pseudopotential, STEP 3) ---
# C.pbesol-sr-rrkjus_psl.1.upf: z_valence=4.0, scalar relativistic, PBESOL, NC
# Recommended ecutwfc = 41.0 Ha (INVARIANT 5: 1 Ha = 2 Ry)
ECUTWFC_HA = 41.0  # Hartree, from Pseudo-Dojo recommended cutoff
ECUTWFC_RY = 2.0 * ECUTWFC_HA  # = 82.0 Ry (INVARIANT 5 conversion)
ECUTRHO_RY = 4.0 * ECUTWFC_RY  # = 328.0 Ry (standard 4x ratio)

# --- Electron count (INVARIANT 2) ---
# 4 C atoms * z_valence 4.0 = 16 total electrons
# Spin-unpolarized: occupied bands = 16/2 = 8
# nbnd for bands: pad above 8, use nbnd=20
NBND = 20

# --- SCF settings ---
# Graphite is a zero-gap semimetal (band_gap_eV=0.0 from MP)
# -> use smearing per occupations RULE
# k-mesh 12x12x4: Monkhorst-Pack, ~1 point per 0.5 A in-plane, 1 per 2 A out-of-plane
SCF_KPTS = (12, 12, 4)

# --- Common settings ---
PREFIX = 'graphite'
OUTDIR = './tmp'

# --- Pseudopotential file ---
PP_FILE = 'C.pbesol-sr-rrkjus_psl.1.upf'
PP_TEMPLATE = os.path.join('template', PP_FILE)

# --- Structures to process ---
N_STRUCTURES = 51  # 000-050

# =============================================================================
# Main loop
# =============================================================================
for nnn in range(N_STRUCTURES):
    nnn_str = f'{nnn:03d}'
    calc_dir = f'calculations/{nnn_str}'
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotential into the calculation directory
    # (pw.x runs inside calculations/NNN/, so pseudo_dir must resolve from there)
    pp_dest = os.path.join(calc_dir, PP_FILE)
    if not os.path.exists(pp_dest):
        shutil.copy2(PP_TEMPLATE, pp_dest)
    
    # Read the structure
    atoms = read(f'structures/structure_{nnn_str}.cif')
    
    # INVARIANT 1: compute nat/ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    
    # INVARIANT 3: get the k-path for THIS structure
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    
    # Save kpath.json (holds labels, ticks, line segments)
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))
    
    # --- Write pw.in (SCF) ---
    # INVARIANT 6: pass a FLAT dict to write_espresso_in; ASE routes each key
    # to its correct namelist (conv_thr -> &ELECTRONS, ecutwfc -> &SYSTEM, etc.)
    scf_input = {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': './',
        'ecutwfc': ECUTWFC_RY,  # Ry
        'ecutrho': ECUTRHO_RY,  # Ry
        'occupations': 'smearing',  # graphite is a semimetal
        'smearing': 'mv',
        'degauss': 0.01,  # Ry
        'conv_thr': 1e-8,  # Ry
    }
    
    # Write SCF input with the primitive cell and k-mesh
    with open(os.path.join(calc_dir, 'pw.in'), 'w') as f:
        write_espresso_in(f, prim, input_data=scf_input,
                          pseudopotentials={'C': PP_FILE},
                          kpts=SCF_KPTS)
    
    # --- Write bands.in (non-self-consistent bands) ---
    bands_input = {
        'calculation': 'bands',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': './',
        'ecutwfc': ECUTWFC_RY,  # Ry
        'ecutrho': ECUTRHO_RY,  # Ry
        'nbnd': NBND,  # padded above 8 occupied bands
    }
    
    # kpts from kp_kpts(kp): shape (n_kpts, 4), crystal coordinates with weights
    bands_kpts = kp_kpts(kp)
    
    with open(os.path.join(calc_dir, 'bands.in'), 'w') as f:
        write_espresso_in(f, prim, input_data=bands_input,
                          pseudopotentials={'C': PP_FILE},
                          kpts=bands_kpts)
    
    # --- Write bands_post.in (bands.x input) ---
    # bands.x reads the SCF wavefunctions from outdir and writes band energies
    # to filband. Use write_fortran_namelist for simple QE binaries.
    from ase.io.espresso import write_fortran_namelist
    bands_post_input = {
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'filband': f'{nnn_str}.bands.dat',
    }
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        write_fortran_namelist(f, input_data=bands_post_input, binary='bands')
    
    if nnn <= 2 or nnn == 50:
        print(f'Wrote {calc_dir}/: pw.in, bands.in, bands_post.in, kpath.json, {PP_FILE}')

print(f'Done: set up {N_STRUCTURES} calculation directories (000-050)')
