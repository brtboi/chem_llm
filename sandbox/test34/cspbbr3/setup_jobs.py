#!/usr/bin/env python3
"""
setup_jobs.py: Create QE input files for all 51 CsPbBr3 structures.

For each structure NNN (000-050), creates calculations/NNN/ containing:
  - Pseudopotential files (copied from template/)
  - pw.in: SCF input (written with ase.io.espresso.write_espresso_in)
  - bands.in: non-self-consistent bands input along seekpath k-path
  - bands_post.in: bands.x post-processing input
  - band_labels.dat: high-symmetry labels and their indices in the k-point list

Key physics decisions:
  - Spin-orbit coupling: Pb (Z=82) > 54, so fully relativistic pseudopotentials
    with noncolin=.true. and lspinorb=.true. in &SYSTEM.
  - With spin-orbit, each band holds ONE electron (not two).
    Total valence electrons = 9 (Cs) + 22 (Pb) + 3*17 (Br) = 83.
    Occupied bands = 83 (not 41.5). nbnd = 100 (padded above 83).
  - ecutwfc: max of recommended cutoffs = 44.0 Ha (Pb).
    INVARIANT 5: 1 Ha = 2 Ry, so ecutwfc = 88.0 Ry.
    ecutrho = 4 * ecutwfc = 352.0 Ry.
  - K-path from seekpath (symprec=0.01) on the primitive cell:
    GAMMA->X->M->GAMMA->R->X->M (6 segments, one discontinuity at X->R)
    20 points per segment (inclusive of both endpoints), deduplicated at
    continuous joints. Total: 115 k-points.
"""

import os
import shutil
import numpy as np
from ase.io import read, write
from ase.io.espresso import write_espresso_in
from ase import Atoms
import json

# --- Constants ---
N_STRUCTURES = 51
NPTS_PER_SEGMENT = 20  # k-points per path segment (inclusive of both endpoints)

# Pseudopotential files (in template/)
PP_FILES = {
    'Cs': 'template/Cs_pbesol_upf.upf',
    'Pb': 'template/Pb_pbesol_upf.upf',
    'Br': 'template/Br_pbesol_upf.upf',
}

# Cutoffs (converted from Ha to Ry per INVARIANT 5: 1 Ha = 2 Ry)
# Max recommended cutoff is Pb: 44.0 Ha -> 88.0 Ry
ECUTWFC_RY = 88.0
ECUTRHO_RY = 4.0 * ECUTWFC_RY  # 352.0 Ry

# Spin-orbit: each band holds 1 electron
# Total valence electrons = 9 + 22 + 3*17 = 83
# Occupied bands = 83, nbnd padded to 100
NBND = 100

# SCF k-point mesh: 6x6x6 Monkhorst-Pack (cubic perovskite, 5 atoms)
SCF_KPTS = (6, 6, 6)

# Common pw.x settings (flat dict -- ASE routes each key to the correct namelist)
COMMON_INPUT = {
    'prefix': 'cspbbr3',
    'outdir': './tmp',
    'pseudo_dir': '.',
    'ecutwfc': ECUTWFC_RY,
    'ecutrho': ECUTRHO_RY,
    'noncolin': True,
    'lspinorb': True,
    'conv_thr': 1.0e-8,
    'occupations': 'fixed',
}


def build_kpath(kpath_info):
    """
    Build the explicit k-point list and high-symmetry labels from seekpath info.
    
    For each segment (from, to) in the path:
      - Interpolate NPTS_PER_SEGMENT points from point_coords[from] to point_coords[to]
      - The first segment includes its start point
      - Subsequent segments: if start == previous end (continuous), skip the start
        (it's already the last point of the previous segment)
      - If start != previous end (discontinuity), include the start (jump)
    
    Returns:
      kpoints: list of (kx, ky, kz) in fractional coordinates of the primitive cell
      labels: list of (index, label) for high-symmetry points
      discontinuities: list of indices where the path jumps
    """
    path = kpath_info['path']  # list of (from, to) tuples
    point_coords = kpath_info['point_coords']  # dict: label -> [x, y, z]
    
    kpoints = []
    labels = []  # (index, label)
    discontinuities = []  # indices where a jump occurs
    
    for seg_idx, (from_label, to_label) in enumerate(path):
        from_pt = np.array(point_coords[from_label])
        to_pt = np.array(point_coords[to_label])
        
        # Interpolate NPTS_PER_SEGMENT points from from_pt to to_pt (inclusive)
        t = np.linspace(0.0, 1.0, NPTS_PER_SEGMENT)
        seg_points = from_pt + t[:, np.newaxis] * (to_pt - from_pt)
        
        if seg_idx == 0:
            # First segment: include all points
            for pt in seg_points:
                kpoints.append(pt)
            # Label the start and end of this segment
            labels.append((0, from_label))
            labels.append((len(kpoints) - 1, to_label))
        else:
            # Check if this segment's start is continuous with the previous end
            prev_end = kpoints[-1]
            is_continuous = np.allclose(from_pt, prev_end, atol=1e-10)
            
            if is_continuous:
                # Skip the start point (already present as previous segment's end)
                for pt in seg_points[1:]:
                    kpoints.append(pt)
                # Label the end of this segment
                labels.append((len(kpoints) - 1, to_label))
            else:
                # Discontinuity: include the start point (jump)
                # Record the discontinuity at the index where the jump happens
                # The jump is between kpoints[-1] (previous end) and from_pt (new start)
                discontinuities.append(len(kpoints))  # index of the new start point
                for pt in seg_points:
                    kpoints.append(pt)
                # Label both sides of the jump
                labels.append((len(kpoints) - NPTS_PER_SEGMENT, from_label))
                labels.append((len(kpoints) - 1, to_label))
    
    return kpoints, labels, discontinuities


def write_band_labels(labels, discontinuities, path):
    """
    Write band_labels.dat with high-symmetry labels and their indices.
    For discontinuities, label the tick with both sides ('X|R').
    """
    # Build a dict: index -> label(s)
    # For discontinuities, the label at the jump index should be 'prev_end|new_start'
    label_dict = {}
    for idx, label in labels:
        if idx in label_dict:
            # Already have a label at this index (shouldn't happen normally)
            label_dict[idx] = f"{label_dict[idx]}|{label}"
        else:
            label_dict[idx] = label
    
    # For discontinuities, the label at the jump index is the new start point.
    # The previous end point's label is at the previous index.
    # We want to label the tick at the jump with 'prev_end|new_start'.
    # The jump index is where the new segment starts.
    # The previous end is at jump_index - 1.
    for jump_idx in discontinuities:
        if jump_idx > 0 and jump_idx in label_dict:
            prev_label = label_dict.get(jump_idx - 1, '?')
            new_label = label_dict[jump_idx]
            # Only combine if they're different (genuine discontinuity)
            if prev_label != new_label:
                label_dict[jump_idx] = f"{prev_label}|{new_label}"
    
    # Sort by index and write
    with open('band_labels.dat', 'w') as f:
        for idx in sorted(label_dict.keys()):
            f.write(f"{idx} {label_dict[idx]}\n")


def main():
    # Load k-path info from seekpath (saved by reduce_primitive.py)
    with open('structures/kpath_info.json', 'r') as f:
        kpath_info = json.load(f)
    
    # Build k-point list and labels
    kpoints, labels, discontinuities = build_kpath(kpath_info)
    n_kpts = len(kpoints)
    print(f"K-path: {kpath_info['path']}")
    print(f"Total k-points: {n_kpts}")
    print(f"High-symmetry labels: {labels}")
    print(f"Discontinuities at indices: {discontinuities}")
    
    # Build the K_POINTS crystal card as a numpy array (n_kpts, 4)
    # Fourth column is weight (unused for bands, set to 1.0)
    kpts_array = np.zeros((n_kpts, 4))
    for i, pt in enumerate(kpoints):
        kpts_array[i, :3] = pt
        kpts_array[i, 3] = 1.0
    
    # Verify k-points are in [0, 1) range (fractional coordinates)
    assert np.all(kpts_array[:, :3] >= -1e-10), "k-point coordinate below 0"
    assert np.all(kpts_array[:, :3] <= 1.0 + 1e-10), "k-point coordinate above 1"
    
    # Process each structure
    for n in range(N_STRUCTURES):
        nnn = f"{n:03d}"
        calc_dir = f"calculations/{nnn}"
        os.makedirs(calc_dir, exist_ok=True)
        os.makedirs(f"{calc_dir}/tmp", exist_ok=True)
        
        # Copy pseudopotential files
        for elem, pp_path in PP_FILES.items():
            shutil.copy2(pp_path, calc_dir)
        
        # Read the structure
        atoms = read(f"structures/structure_{nnn}.cif")
        
        # INVARIANT 1: nat and ntyp from the actual Atoms object
        nat = len(atoms)
        ntyp = len(set(atoms.get_chemical_symbols()))
        assert nat == 5, f"Structure {nnn}: expected nat=5, got {nat}"
        assert ntyp == 3, f"Structure {nnn}: expected ntyp=3, got {ntyp}"
        
        # Pseudopotential filenames (relative to calc_dir, where pw.x runs)
        pseudopotentials = {
            'Cs': 'Cs_pbesol_upf.upf',
            'Pb': 'Pb_pbesol_upf.upf',
            'Br': 'Br_pbesol_upf.upf',
        }
        
        # --- SCF input (pw.in) ---
        scf_input = dict(COMMON_INPUT)
        scf_input['calculation'] = 'scf'
        
        write_espresso_in(
            f"{calc_dir}/pw.in",
            atoms,
            input_data=scf_input,
            pseudopotentials=pseudopotentials,
            kpts=SCF_KPTS,
            crystal_coordinates=True,
        )
        
        # --- Bands input (bands.in) ---
        bands_input = dict(COMMON_INPUT)
        bands_input['calculation'] = 'bands'
        bands_input['nbnd'] = NBND
        
        write_espresso_in(
            f"{calc_dir}/bands.in",
            atoms,
            input_data=bands_input,
            pseudopotentials=pseudopotentials,
            kpts=kpts_array,  # explicit k-point list (n_kpts, 4)
            crystal_coordinates=True,
        )
        
        # --- Bands post-processing input (bands_post.in) ---
        # bands.x input: &BANDS namelist with prefix, outdir, filband
        with open(f"{calc_dir}/bands_post.in", 'w') as f:
            f.write("&BANDS\n")
            f.write(f"  prefix = '{COMMON_INPUT['prefix']}'\n")
            f.write(f"  outdir = '{COMMON_INPUT['outdir']}'\n")
            f.write(f"  filband = '{nnn}.bands.dat'\n")
            f.write("/\n")
            f.write("EOF\n")
        
        # --- Band labels ---
        # Write to calc_dir (plot_bands.py will read from there)
        label_dict = {}
        for idx, label in labels:
            if idx in label_dict:
                label_dict[idx] = f"{label_dict[idx]}|{label}"
            else:
                label_dict[idx] = label
        
        for jump_idx in discontinuities:
            if jump_idx > 0 and jump_idx in label_dict:
                prev_label = label_dict.get(jump_idx - 1, '?')
                new_label = label_dict[jump_idx]
                if prev_label != new_label:
                    label_dict[jump_idx] = f"{prev_label}|{new_label}"
        
        with open(f"{calc_dir}/band_labels.dat", 'w') as f:
            for idx in sorted(label_dict.keys()):
                f.write(f"{idx} {label_dict[idx]}\n")
        
        if n < 3 or n == N_STRUCTURES - 1:
            print(f"Wrote {calc_dir}/: pw.in, bands.in, bands_post.in, band_labels.dat")
    
    print(f"\nDone. Set up {N_STRUCTURES} calculation directories in calculations/")
    print(f"ecutwfc = {ECUTWFC_RY} Ry, ecutrho = {ECUTRHO_RY} Ry")
    print(f"nbnd = {NBND}, SCF kpts = {SCF_KPTS}")
    print(f"Bands k-points: {n_kpts} along path {kpath_info['path']}")


if __name__ == '__main__':
    main()
