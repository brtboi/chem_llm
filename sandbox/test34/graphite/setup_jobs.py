import numpy as np
import os
import shutil
import seekpath
import spglib
from ase.io import read, write
from ase.io.espresso import write_espresso_in
from ase import Atoms

# =============================================================================
# setup_jobs.py
# For each structure NNN (000-050), create calculations/NNN/ containing:
#   - Pseudopotential files (copied from template/)
#   - pw.in (SCF input)
#   - bands.in (non-self-consistent bands along k-path)
#   - bands_post.in (bands.x post-processing input)
#   - band_labels.dat (high-symmetry labels and their indices)
# =============================================================================

# --- Pseudopotential info (from STEP 3) ---
PP_FILE = 'C.pbesol-sr-rrkjus_psl.1.0.0.UPF'
PP_DIR = 'template'

# --- Cutoffs (INVARIANT 5: convert from Ha to Ry) ---
# Recommended cutoff from Pseudo-Dojo: 41.0 Ha
# 1 Ha = 2 Ry
ECUTWFC_HA = 41.0
ECUTWFC_RY = 2.0 * ECUTWFC_HA  # 82.0 Ry
ECUTRHO_RY = 4.0 * ECUTWFC_RY  # 328.0 Ry (4x ratio for local part)

# --- SCF settings ---
PREFIX = 'graphite'
OUTDIR = './tmp'
CONV_THR = 1e-8

# K-point mesh for SCF: 12x12x4 Monkhorst-Pack
# Denser in-plane (12x12) due to 2D character of graphite,
# sparser along c (4) since interlayer coupling is weak
KPTS_SCF = (12, 12, 4)

# --- Bands settings ---
# nbnd: 4 occupied bands (8 electrons / 2) + 8 unoccupied = 12 total
# Enough to see the gap and several conduction bands
NBND = 12

# Points per segment for k-path interpolation
PTS_PER_SEG = 20

# --- Number of structures ---
N_STRUCTURES = 51

# --- Pseudopotentials dict for write_espresso_in ---
# The filename must be relative to pseudo_dir (which is './' = calculations/NNN/)
# So we copy the PP file into each calculations/NNN/ directory
PSEUDOS = {'C': PP_FILE}

# =============================================================================
# Helper: build k-path from seekpath for a given Atoms object
# =============================================================================
def build_kpath(atoms):
    """
    Build the k-path from seekpath for the given Atoms object.
    Returns:
      - prim: the primitive Atoms object (from seekpath) to use in CELL_PARAMETERS
      - kpts_array: (n_kpts, 4) array in crystal coordinates for the primitive cell
      - labels: list of (label, index) for each high-symmetry point in the path
      - path_segments: list of (label_from, label_to) segments
    """
    cell_tuple = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    res = seekpath.get_path(cell_tuple, symprec=0.01)
    
    # Verify space group consistency (INVARIANT 3)
    spg_dataset = spglib.get_symmetry_dataset(cell_tuple, symprec=0.01)
    assert res['spacegroup_number'] == spg_dataset.number, \
        f"Space group mismatch: seekpath={res['spacegroup_number']}, spglib={spg_dataset.number}"
    
    # Build the primitive Atoms object (this is what goes into CELL_PARAMETERS)
    prim = Atoms(
        cell=res['primitive_lattice'],
        scaled_positions=res['primitive_positions'],
        numbers=res['primitive_types'],
        pbc=True
    )
    
    # Build the k-point list by interpolating each segment
    # Each segment contributes its own start point + interior points
    # (INVARIANT 3: path is frequently discontinuous)
    path = res['path']  # list of (label_from, label_to)
    point_coords = res['point_coords']  # dict of label -> fractional coords
    
    kpts_list = []  # list of (kx, ky, kz, weight)
    labels = []  # list of (label, index_in_kpts_list)
    
    for i_seg, (label_from, label_to) in enumerate(path):
        k_from = np.array(point_coords[label_from])
        k_to = np.array(point_coords[label_to])
        
        # Add the start point of this segment
        # (For the first segment, this is the very first point)
        # For subsequent segments, this handles discontinuities
        kpts_list.append((k_from[0], k_from[1], k_from[2], 1.0))
        
        # Record the label at this point
        # If this is a discontinuity (previous segment's end != this segment's start),
        # we'll handle the label in the plotting step
        labels.append((label_from, len(kpts_list) - 1))
        
        # Add interior points (excluding the start, including the end)
        for i in range(1, PTS_PER_SEG + 1):
            t = i / PTS_PER_SEG
            k = (1 - t) * k_from + t * k_to
            kpts_list.append((k[0], k[1], k[2], 1.0))
        
        # Record the end label
        labels.append((label_to, len(kpts_list) - 1))
    
    kpts_array = np.array(kpts_list)
    
    return prim, kpts_array, labels, path

# =============================================================================
# Main loop over structures
# =============================================================================
for i in range(N_STRUCTURES):
    nnn = f"{i:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotential file into the calculation directory
    pp_src = os.path.join(PP_DIR, PP_FILE)
    pp_dst = os.path.join(calc_dir, PP_FILE)
    if not os.path.exists(pp_dst):
        shutil.copy2(pp_src, pp_dst)
    
    # Read the structure
    atoms = read(f"structures/structure_{nnn}.cif")
    
    # INVARIANT 1: compute nat/ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    
    # Build k-path from THIS structure (INVARIANT 3)
    prim, kpts_array, labels, path_segments = build_kpath(atoms)
    
    # The primitive cell is what goes into CELL_PARAMETERS
    # For structure 000 (unperturbed), prim should be identical to atoms
    # For perturbed structures, prim is the symmetry-reduced version
    # We use prim for the QE input since the k-path coordinates refer to it
    
    # --- Write SCF input (pw.in) ---
    scf_input_data = {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': './',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
        'nat': nat,
        'ntyp': ntyp,
    }
    
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=scf_input_data,
        pseudopotentials=PSEUDOS,
        kpts=KPTS_SCF,
    )
    
    # --- Write bands input (bands.in) ---
    bands_input_data = {
        'calculation': 'bands',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': './',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'nbnd': NBND,
        'nat': nat,
        'ntyp': ntyp,
    }
    
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=bands_input_data,
        pseudopotentials=PSEUDOS,
        kpts=kpts_array,
    )
    
    # --- Write bands.x post-processing input (bands_post.in) ---
    # bands.x uses a simple namelist format
    bands_post_content = f"""&BANDS
  prefix = '{PREFIX}'
  outdir = '{OUTDIR}'
  filband = '{nnn}.bands.dat'
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)
    
    # --- Write band_labels.dat ---
    # Format: one line per high-symmetry point: label index
    with open(os.path.join(calc_dir, 'band_labels.dat'), 'w') as f:
        for label, idx in labels:
            f.write(f"{label} {idx}\n")
    
    print(f"Wrote {calc_dir}/: pw.in, bands.in, bands_post.in, band_labels.dat")
    print(f"  nat={nat}, ntyp={ntyp}, n_kpts={len(kpts_array)}, nbnd={NBND}")

print(f"\nDone. Set up {N_STRUCTURES} calculation directories.")
