#!/usr/bin/env python3
"""
setup_jobs.py: Create QE input files for each structure NNN (000-050).

For each structure, creates calculations/NNN/ containing:
  - Ti.pbesol_upf, O.pbesol_upf (pseudopotentials copied from template/)
  - pw.in (SCF input, written with write_espresso_in)
  - bands.in (bands input, written with write_espresso_in)
  - bands_post.in (bands.x input, written with write_fortran_namelist)
  - band_labels.dat (high-symmetry labels and their 0-based indices in the k-point list)

K-path: derived from seekpath on the EXACT Atoms object written into CELL_PARAMETERS.
seekpath's primitive cell is used as the cell for the calculation, so its
point_coords (fractional in that cell) apply directly to the K_POINTS card.

pw.x settings (all units noted):
  SCF:
    calculation = 'scf'
    prefix = 'tio2'
    outdir = './tmp' (relative to calculations/NNN/)
    pseudo_dir = '.' (pseudopotentials are in the same directory)
    ecutwfc = 92.0 Ry  (46.0 Ha * 2, INVARIANT 5: 1 Ha = 2 Ry)
    ecutrho = 368.0 Ry (4 * ecutwfc, standard 4x ratio for charge density)
    occupations = 'fixed' (insulator, no smearing needed)
    conv_thr = 1.0d-8 (in &ELECTRONS, routed by write_espresso_in, INVARIANT 6)
    k-mesh = 4x4x4 Monkhorst-Pack (adequate for 6-atom tetragonal cell)
  Bands:
    calculation = 'bands'
    Same prefix/outdir/pseudo_dir/ecutwfc/ecutrho
    nbnd = 48 (2x occupied bands of 24; total electrons = 2*12 + 4*6 = 48,
               occupied = 24 for spin-unpolarized non-metallic run)
    K_POINTS: crystal card, explicit (n_kpts, 4) array from seekpath path,
               weights = 1.0 (unused for non-self-consistent bands run)

bands.x settings:
  &BANDS: prefix='tio2', outdir='./tmp', filband='NNN.bands.dat'
"""

import numpy as np
import os
import shutil
import seekpath
import spglib
from ase.io import read, write
from ase.io.espresso import write_espresso_in, write_fortran_namelist
from ase import Atoms

# --- Constants ---
N_STRUCTURES = 51  # 000 through 050
TEMPLATE_DIR = 'template'
STRUCTURES_DIR = 'structures'
CALC_DIR = 'calculations'

# Pseudopotential files
PP_Ti = 'Ti.pbesol_upf'
PP_O = 'O.pbesol_upf'

# Cutoffs (Ry) -- INVARIANT 5: 1 Ha = 2 Ry
# Ti recommended: 42.0 Ha, O recommended: 46.0 Ha
# Use max = 46.0 Ha = 92.0 Ry
ECUTWFC_RY = 92.0
ECUTRHO_RY = 4.0 * ECUTWFC_RY  # 368.0 Ry

# nbnd: total electrons = 2*12 (Ti) + 4*6 (O) = 48, occupied = 24
# nbnd = 48 (2x occupied, safe margin)
NBND = 48

# SCF k-mesh
KPTS_SCF = (4, 4, 4)

# Points per segment in the k-path
PTS_PER_SEG = 20


def get_kpath(atoms):
    """
    Derive the k-path from the EXACT Atoms object using seekpath.
    Returns:
      prim: Atoms object (seekpath's primitive cell) to use in CELL_PARAMETERS
      kpts: numpy array (n_kpts, 4) in crystal coordinates of prim's cell
      labels: list of (label, index) for band_labels.dat (0-based indices)
      path: list of (label_from, label_to) segments
    """
    cell_tuple = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    
    # Verify space group consistency (INVARIANT 3)
    dataset = spglib.get_symmetry_dataset(cell_tuple, symprec=0.01)
    res = seekpath.get_path(cell_tuple, symprec=0.01)
    assert res['spacegroup_number'] == dataset.number, \
        f"seekpath ({res['spacegroup_number']}) != spglib ({dataset.number})"
    
    # Build the primitive cell from seekpath's own output
    prim = Atoms(
        cell=res['primitive_lattice'],
        scaled_positions=res['primitive_positions'],
        numbers=res['primitive_types'],
        pbc=True
    )
    
    # res['point_coords'] is a dict: label -> fractional coordinates
    # (in the primitive cell's reciprocal lattice)
    point_coords = res['point_coords']
    path = res['path']  # list of (label_from, label_to)
    
    # Build the explicit k-point list.
    # INVARIANT 3: EVERY segment must contribute its own start point.
    # For a continuous path (prev_end == this_start), the start point is the
    # same as the previous segment's end, so we do NOT re-append it -- we just
    # continue interpolating from it. For a discontinuity, we append the start
    # point as a new k-point.
    #
    # Labels are recorded at the 0-based index of the k-point where the
    # high-symmetry point sits. At a discontinuity, the label is 'PREV|NEXT'
    # (both sides) at the NEW start point's index.
    
    kpts_list = []   # list of (kx, ky, kz) in fractional coords of prim's cell
    labels = []      # list of (label, 0-based index) for band_labels.dat
    
    for i_seg, (lbl_from, lbl_to) in enumerate(path):
        coord_from = np.array(point_coords[lbl_from])
        coord_to = np.array(point_coords[lbl_to])
        
        if i_seg == 0:
            # First segment: append start point, then interior points
            start_idx = len(kpts_list)
            kpts_list.append(coord_from)
            labels.append((lbl_from, start_idx))
        else:
            # Check if this is a discontinuity
            prev_lbl_to = path[i_seg - 1][1]
            prev_coord_to = np.array(point_coords[prev_lbl_to])
            
            if np.allclose(prev_coord_to, coord_from, atol=1e-10):
                # Continuous: the start point is already the last point in
                # kpts_list (the previous segment's end). Do NOT re-append.
                # Record the label at the shared index (0-based).
                start_idx = len(kpts_list) - 1
                labels.append((lbl_from, start_idx))
            else:
                # Discontinuity: append the start point as a new k-point
                start_idx = len(kpts_list)
                kpts_list.append(coord_from)
                # Label with both sides: 'PREV|NEXT'
                labels.append((f"{prev_lbl_to}|{lbl_from}", start_idx))
        
        # Interior points (excluding start, including end)
        for i in range(1, PTS_PER_SEG + 1):
            t = i / PTS_PER_SEG
            k = (1 - t) * coord_from + t * coord_to
            kpts_list.append(k)
        
        # Record the end label of this segment
        end_idx = len(kpts_list) - 1
        if i_seg == len(path) - 1:
            # Last segment: always record the end label
            labels.append((lbl_to, end_idx))
        else:
            # Check if the next segment starts at the same point
            next_lbl_from = path[i_seg + 1][0]
            next_coord_from = np.array(point_coords[next_lbl_from])
            if not np.allclose(coord_to, next_coord_from, atol=1e-10):
                # Discontinuity: record the end label (the next start will be
                # recorded separately with a combined label)
                labels.append((lbl_to, end_idx))
            # If continuous, the end label will be recorded as the start label
            # of the next segment (handled in the next iteration)
    
    # Sort labels by index (they should already be in order, but just in case)
    labels.sort(key=lambda x: x[1])
    
    # Convert to numpy array (n_kpts, 4) with weights = 1.0
    kpts_array = np.zeros((len(kpts_list), 4))
    for i, k in enumerate(kpts_list):
        kpts_array[i, :3] = k
        kpts_array[i, 3] = 1.0  # weight (unused for bands run)
    
    return prim, kpts_array, labels, path


def write_pw_in(atoms, fname, calculation, kpts, nbnd=None):
    """
    Write a pw.x input file using write_espresso_in.
    
    Args:
      atoms: Atoms object (the structure to write into CELL_PARAMETERS)
      fname: output filename
      calculation: 'scf' or 'bands'
      kpts: for SCF, a tuple (nx, ny, nz); for bands, a numpy array (n_kpts, 4)
      nbnd: number of bands (required for bands calculation)
    """
    # Build the flat input_data dict (INVARIANT 6: let ASE route to correct namelists)
    input_data = {
        'control': {
            'calculation': calculation,
            'prefix': 'tio2',
            'outdir': './tmp',
            'pseudo_dir': '.',  # pseudopotentials are in the same directory
        },
        'system': {
            'ecutwfc': ECUTWFC_RY,
            'ecutrho': ECUTRHO_RY,
            'occupations': 'fixed',
        },
        'electrons': {
            'conv_thr': 1.0e-8,
        },
    }
    
    if calculation == 'bands':
        input_data['system']['nbnd'] = nbnd
    
    # Pseudopotentials: map element -> filename
    pseudopotentials = {
        'Ti': PP_Ti,
        'O': PP_O,
    }
    
    # Write the input file
    with open(fname, 'w') as f:
        write_espresso_in(
            f,
            atoms,
            input_data=input_data,
            pseudopotentials=pseudopotentials,
            kpts=kpts,
            crystal_coordinates=True,
        )


def write_bands_post_in(fname, nnn):
    """
    Write the bands.x input file (bands_post.in).
    """
    input_data = {
        'BANDS': {
            'prefix': 'tio2',
            'outdir': './tmp',
            'filband': f'{nnn}.bands.dat',
        }
    }
    with open(fname, 'w') as f:
        write_fortran_namelist(f, input_data, binary='bands')


def write_band_labels(fname, labels):
    """
    Write band_labels.dat: each line is 'label index' where index is the
    0-based index into the k-point list (as expected by validate_qe_input).
    """
    with open(fname, 'w') as f:
        for label, idx in labels:
            # idx is already 0-based; write it as-is
            f.write(f"{label} {idx}\n")


def main():
    os.makedirs(CALC_DIR, exist_ok=True)
    
    for nnn in range(N_STRUCTURES):
        nnn_str = f"{nnn:03d}"
        calc_path = os.path.join(CALC_DIR, nnn_str)
        os.makedirs(calc_path, exist_ok=True)
        
        # Copy pseudopotentials
        for pp in [PP_Ti, PP_O]:
            src = os.path.join(TEMPLATE_DIR, pp)
            dst = os.path.join(calc_path, pp)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        
        # Read the structure
        cif_path = os.path.join(STRUCTURES_DIR, f'structure_{nnn_str}.cif')
        atoms = read(cif_path)
        
        # Verify nat/ntyp INVARIANT (INVARIANT 1)
        nat = len(atoms)
        ntyp = len(set(atoms.get_chemical_symbols()))
        assert nat == 6, f"Structure {nnn_str}: expected 6 atoms, got {nat}"
        assert ntyp == 2, f"Structure {nnn_str}: expected 2 element types, got {ntyp}"
        
        # Derive the k-path from THIS structure (INVARIANT 3)
        prim, kpts_array, labels, path = get_kpath(atoms)
        
        # Verify the primitive cell has the right atom count
        assert len(prim) == 6, \
            f"Structure {nnn_str}: primitive cell has {len(prim)} atoms, expected 6"
        
        # Write pw.in (SCF) using the primitive cell
        pw_in_path = os.path.join(calc_path, 'pw.in')
        write_pw_in(prim, pw_in_path, calculation='scf', kpts=KPTS_SCF)
        
        # Write bands.in using the primitive cell and the k-path
        bands_in_path = os.path.join(calc_path, 'bands.in')
        write_pw_in(prim, bands_in_path, calculation='bands',
                    kpts=kpts_array, nbnd=NBND)
        
        # Write bands_post.in
        bands_post_path = os.path.join(calc_path, 'bands_post.in')
        write_bands_post_in(bands_post_path, nnn_str)
        
        # Write band_labels.dat
        band_labels_path = os.path.join(calc_path, 'band_labels.dat')
        write_band_labels(band_labels_path, labels)
        
        if nnn % 10 == 0:
            print(f"Setup complete for structure {nnn_str}: "
                  f"{len(kpts_array)} k-points, {len(labels)} labels")
    
    print(f"\nDone. Set up {N_STRUCTURES} calculation directories in {CALC_DIR}/")


if __name__ == '__main__':
    main()
