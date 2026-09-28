#!/usr/bin/env python3
"""
Setup QE calculation directories for all 51 rutile TiO2 structures.

For each structure NNN, creates calculations/NNN/ containing:
  - Pseudopotential files (copied from template/)
  - pw.in (SCF input, written with write_espresso_in)
  - bands.in (bands input, written with write_espresso_in)
  - bands_post.in (bands.x input, written with write_fortran_namelist)
  - band_labels.dat (high-symmetry labels and their indices in the k-point list)

INVARIANT 1: nat/ntyp computed from actual Atoms object.
INVARIANT 3: k-path derived from seekpath with symprec=0.01, reprojected to actual cell reciprocal.
INVARIANT 5: ecutwfc converted from Ha to Ry (1 Ha = 2 Ry).
INVARIANT 6: flat input_data dict passed to write_espresso_in for correct namelist routing.
"""

import os
import shutil
import numpy as np
import spglib
import seekpath
from ase.io import read, write
from ase.io.espresso import write_espresso_in, write_fortran_namelist

# --- Constants ---
# Pseudopotential cutoff: 42.0 Ha from get_pseudopotential hint
# INVARIANT 5: 1 Ha = 2 Ry, so ecutwfc = 84.0 Ry
ECUTWFC_RY = 84.0
ECUTRHO_RY = 4.0 * ECUTWFC_RY  # 336.0 Ry
CONV_THR = 1e-8  # Ry
NBND = 40  # Generous; will verify against pw.out after SCF

# Pseudopotential files (in template/)
PP_Ti = 'template/Ti.pbesol-n-rrkjus_psl.1.0.0.UPF'
PP_O = 'template/O.pbesol-n-rrkjus_psl.1.0.0.UPF'

# K-points per segment for the bands path
NPTS_PER_SEGMENT = 20


def get_kpath(atoms, npts_per_segment=20):
    """
    Derive the k-path for a band structure calculation using seekpath.
    
    INVARIANT 3: Use seekpath with symprec=0.01, reproject to actual cell reciprocal
    via Cartesian coordinates. Every segment contributes its own start point.
    
    Returns:
        kpts: (n_kpts, 4) array in crystal coordinates (fractional)
        labels: list of (label, index) tuples for high-symmetry points
        path_segments: list of (label_from, label_to) for discontinuity handling
    """
    cell_tuple = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    
    # Get the path from seekpath with symprec=0.01 (matching pymatgen default)
    res = seekpath.get_path(cell_tuple, symprec=0.01)
    
    # Verify space group consistency with spglib
    dataset = spglib.get_symmetry_dataset(cell_tuple, symprec=0.01)
    assert res['spacegroup_number'] == dataset.number, \
        f"Space group mismatch: seekpath={res['spacegroup_number']}, spglib={dataset.number}"
    
    # Reciprocal lattices
    # seekpath's reciprocal_primitive_lattice already includes 2*pi
    recip_prim = np.array(res['reciprocal_primitive_lattice'])
    # ASE's cell.reciprocal() does NOT include 2*pi -- must add it
    recip_actual = 2 * np.pi * np.array(atoms.cell.reciprocal()[:])
    
    def to_actual_frac(frac_prim):
        """Convert fractional coords in seekpath's primitive reciprocal lattice
        to fractional coords in the actual cell's reciprocal lattice."""
        cart = np.array(frac_prim) @ recip_prim
        return cart @ np.linalg.inv(recip_actual)
    
    # Convert all special point coordinates to actual cell fractional coords
    point_coords_actual = {lbl: to_actual_frac(c) for lbl, c in res['point_coords'].items()}
    
    # Build the k-point list: each segment contributes its own start point
    # plus interior points (INVARIANT 3: path is frequently discontinuous)
    kpts_list = []
    labels = []  # (label, index) for high-symmetry points
    path_segments = []  # (label_from, label_to) for each segment
    
    for i, (lbl_from, lbl_to) in enumerate(res['path']):
        path_segments.append((lbl_from, lbl_to))
        start = point_coords_actual[lbl_from]
        end = point_coords_actual[lbl_to]
        
        # Add the start point of this segment
        if i == 0:
            kpts_list.append(start)
            labels.append((lbl_from, 0))
        else:
            # For discontinuous paths, the start point of a new segment is a jump
            # Record both labels at the jump
            kpts_list.append(start)
            labels.append((f"{lbl_from}|{lbl_to}", len(kpts_list) - 1))
        
        # Add interior points (excluding the start, including the end)
        for j in range(1, npts_per_segment + 1):
            t = j / npts_per_segment
            kpt = (1 - t) * start + t * end
            kpts_list.append(kpt)
        
        # Record the end point label (only if it's not the start of the next segment)
        if i < len(res['path']) - 1:
            next_lbl_from = res['path'][i + 1][0]
            if lbl_to != next_lbl_from:
                # This is a discontinuity: the end of this segment is not the start of the next
                labels.append((lbl_to, len(kpts_list) - 1))
        else:
            # Last segment: record the end point
            labels.append((lbl_to, len(kpts_list) - 1))
    
    # Build the (n_kpts, 4) array: 3 fractional coords + weight (1.0, unused for bands)
    kpts = np.zeros((len(kpts_list), 4))
    for i, kpt in enumerate(kpts_list):
        kpts[i, :3] = kpt
        kpts[i, 3] = 1.0
    
    return kpts, labels, path_segments


def setup_structure(nnn, atoms):
    """Set up the calculation directory for one structure."""
    calc_dir = f'calculations/{nnn}'
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotentials into the calculation directory
    # (pseudo_dir = './' so pw.x looks in the current directory)
    shutil.copy(PP_Ti, calc_dir)
    shutil.copy(PP_O, calc_dir)
    
    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    
    # Pseudopotential mapping
    pseudopotentials = {
        'Ti': os.path.basename(PP_Ti),
        'O': os.path.basename(PP_O),
    }
    
    # --- SCF input (pw.in) ---
    scf_input_data = {
        'calculation': 'scf',
        'prefix': f'calc{nnn}',
        'outdir': './tmp',
        'pseudo_dir': './',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
        'nat': nat,
        'ntyp': ntyp,
    }
    
    # 4x4x4 Monkhorst-Pack grid for SCF
    kpts_scf = (4, 4, 4)
    
    with open(os.path.join(calc_dir, 'pw.in'), 'w') as f:
        write_espresso_in(f, atoms, input_data=scf_input_data,
                          pseudopotentials=pseudopotentials, kpts=kpts_scf)
    
    # --- Bands input (bands.in) ---
    # Derive k-path from this specific structure (INVARIANT 3)
    kpts_bands, labels, path_segments = get_kpath(atoms, NPTS_PER_SEGMENT)
    
    bands_input_data = {
        'calculation': 'bands',
        'prefix': f'calc{nnn}',
        'outdir': './tmp',
        'pseudo_dir': './',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
        'nbnd': NBND,
        'nat': nat,
        'ntyp': ntyp,
    }
    
    with open(os.path.join(calc_dir, 'bands.in'), 'w') as f:
        write_espresso_in(f, atoms, input_data=bands_input_data,
                          pseudopotentials=pseudopotentials, kpts=kpts_bands)
    
    # --- Bands post-processing input (bands_post.in) ---
    bands_post_input_data = {
        'prefix': f'calc{nnn}',
        'outdir': './tmp',
        'filband': f'{nnn}.bands.dat',
    }
    
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        write_fortran_namelist(f, input_data=bands_post_input_data, binary='bands')
    
    # --- Band labels file (for plotting) ---
    with open(os.path.join(calc_dir, 'band_labels.dat'), 'w') as f:
        f.write(f"# Band labels for structure {nnn}\n")
        f.write(f"# k-point count: {len(kpts_bands)}\n")
        f.write(f"# Path segments: {path_segments}\n")
        for lbl, idx in labels:
            f.write(f"{lbl}\t{idx}\n")
    
    return kpts_bands, labels, path_segments


def main():
    print("Setting up calculation directories for 51 structures...")
    
    for nnn in range(51):
        cif_path = f'structures/structure_{nnn:03d}.cif'
        atoms = read(cif_path)
        
        kpts, labels, segments = setup_structure(f'{nnn:03d}', atoms)
        
        if nnn == 0:
            print(f"Structure 000: {len(atoms)} atoms, {len(set(atoms.get_chemical_symbols()))} types")
            print(f"  k-path: {len(kpts)} points, {len(labels)} labels")
            print(f"  Path segments: {segments}")
            print(f"  Labels: {labels}")
        elif nnn % 10 == 0:
            print(f"Structure {nnn:03d}: {len(atoms)} atoms, k-path: {len(kpts)} points")
    
    print("\nDone: all 51 calculation directories set up.")


if __name__ == '__main__':
    main()
