#!/usr/bin/env python3
"""
setup_jobs.py: For each structure NNN (000-050), create calculations/NNN/ containing:
- Pseudopotential files (Ti.upf, O.upf) copied from template/
- pw.in (SCF input) written with ase.io.espresso.write_espresso_in
- bands.in (non-self-consistent bands input) written with write_espresso_in
- bands_post.in (bands.x input with &BANDS namelist)
- band_labels.dat (high-symmetry labels and their indices in the k-point list)

Key decisions (from STEP 4 note):
- ecutwfc = 92.0 Ry (max of Ti 42 Ha, O 46 Ha; 1 Ha = 2 Ry, INVARIANT 5)
- ecutrho = 368.0 Ry (4x ecutwfc)
- occupations = 'fixed' (insulator)
- conv_thr = 1e-8 Ry
- SCF k-mesh: 4x4x4 Monkhorst-Pack
- nbnd = 40 (occupied bands = 24, from 48 total electrons / 2)
- K-path: seekpath-derived, 20 points per segment, discontinuities at Z->X and R->M
"""

import os
import shutil
import numpy as np
import seekpath
import spglib
from ase.io import read, write
from ase.io.espresso import write_espresso_in
from ase import Atoms

# Pseudopotential files
PSEUDO_DIR = 'template'
Ti_pseudo = 'Ti.upf'
O_pseudo = 'O.upf'

# Cutoffs (converted from Ha to Ry: 1 Ha = 2 Ry, INVARIANT 5)
# Ti: 42.0 Ha -> 84.0 Ry
# O: 46.0 Ha -> 92.0 Ry
# Use the maximum: 92.0 Ry
ECUTWFC_RY = 92.0
ECUTRHO_RY = 4.0 * ECUTWFC_RY  # 368.0 Ry

# SCF settings
CONV_THR = 1.0e-8  # Ry
KPTS_SCF = (4, 4, 4)  # Monkhorst-Pack grid

# Bands settings
NBND = 40  # occupied bands = 24 (48 electrons / 2), well above
NPTS_PER_SEGMENT = 20  # k-points per segment (including endpoints)

# Prefix and outdir
PREFIX = 'tio2'
OUTDIR = './tmp'


def get_kpath(atoms):
    """
    Derive the k-path from the exact ASE Atoms object using seekpath.
    Returns:
        prim: Atoms object for the primitive cell (to use in CELL_PARAMETERS)
        kpoints: (n_kpts, 4) array in crystal coordinates for the K_POINTS card
        labels: list of (label, index) for band_labels.dat
        path: list of (label_from, label_to) segments
    """
    cell_tuple = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    res = seekpath.get_path(cell_tuple, symprec=0.01)

    # Verify space group consistency
    spg_dataset = spglib.get_symmetry_dataset(cell_tuple, symprec=0.01)
    assert res['spacegroup_number'] == spg_dataset.number, \
        f"Space group mismatch: seekpath={res['spacegroup_number']}, spglib={spg_dataset.number}"

    # Build the primitive cell from seekpath's result
    prim = Atoms(
        cell=res['primitive_lattice'],
        scaled_positions=res['primitive_positions'],
        numbers=res['primitive_types'],
        pbc=True
    )

    # Build the k-point list from the path
    # Each segment: (label_from, label_to)
    # For continuous segments, skip the duplicate start point
    # For discontinuous segments, include the start point
    path = res['path']
    point_coords = res['point_coords']  # dict keyed by label

    kpoints = []
    labels = []  # (label, index) for high-symmetry points

    for i_seg, (lbl_from, lbl_to) in enumerate(path):
        coord_from = np.array(point_coords[lbl_from], dtype=float)
        coord_to = np.array(point_coords[lbl_to], dtype=float)

        # Generate NPTS_PER_SEGMENT points from start to end (inclusive)
        t = np.linspace(0, 1, NPTS_PER_SEGMENT)
        seg_points = coord_from[None, :] + t[:, None] * (coord_to - coord_from)[None, :]

        # Determine if this segment is continuous with the previous one
        if i_seg == 0:
            # First segment: include all points
            for pt in seg_points:
                kpoints.append(pt)
            # Record the start label at index 0
            labels.append((lbl_from, 0))
            # Record the end label at the last index of this segment
            labels.append((lbl_to, len(kpoints) - 1))
        else:
            # Check if continuous: previous segment's end == this segment's start
            prev_lbl_to = path[i_seg - 1][1]
            prev_coord_to = np.array(point_coords[prev_lbl_to], dtype=float)
            is_continuous = np.allclose(prev_coord_to, coord_from, atol=1e-10)

            if is_continuous:
                # Skip the start point (it's the same as the previous end)
                for pt in seg_points[1:]:
                    kpoints.append(pt)
            else:
                # Discontinuous: include all points
                for pt in seg_points:
                    kpoints.append(pt)

            # Record the start label (only if discontinuous)
            if not is_continuous:
                # The start point is at index len(kpoints) - NPTS_PER_SEGMENT
                start_idx = len(kpoints) - NPTS_PER_SEGMENT
                # Label with both sides: "prev_lbl_to|lbl_from"
                labels.append((f"{prev_lbl_to}|{lbl_from}", start_idx))

            # Record the end label at the last index of this segment
            labels.append((lbl_to, len(kpoints) - 1))

    kpoints = np.array(kpoints)
    # Add weight column (constant 1.0, unused for non-self-consistent bands)
    weights = np.ones((len(kpoints), 1))
    kpoints_full = np.hstack([kpoints, weights])

    return prim, kpoints_full, labels, path


def write_pw_in(atoms, out_path, kpts):
    """Write SCF input file using write_espresso_in."""
    input_data = {
        'control': {
            'calculation': 'scf',
            'prefix': PREFIX,
            'outdir': OUTDIR,
            'pseudo_dir': './',
        },
        'system': {
            'ecutwfc': ECUTWFC_RY,
            'ecutrho': ECUTRHO_RY,
            'occupations': 'fixed',
        },
        'electrons': {
            'conv_thr': CONV_THR,
        },
    }
    pseudopotentials = {'Ti': Ti_pseudo, 'O': O_pseudo}
    write_espresso_in(out_path, atoms, input_data=input_data,
                      pseudopotentials=pseudopotentials, kpts=kpts)


def write_bands_in(atoms, out_path, kpoints):
    """Write bands input file using write_espresso_in."""
    input_data = {
        'control': {
            'calculation': 'bands',
            'prefix': PREFIX,
            'outdir': OUTDIR,
            'pseudo_dir': './',
        },
        'system': {
            'ecutwfc': ECUTWFC_RY,
            'ecutrho': ECUTRHO_RY,
            'nbnd': NBND,
        },
    }
    pseudopotentials = {'Ti': Ti_pseudo, 'O': O_pseudo}
    # kpoints is (n_kpts, 4) array in crystal coordinates
    write_espresso_in(out_path, atoms, input_data=input_data,
                      pseudopotentials=pseudopotentials, kpts=kpoints)


def write_bands_post_in(out_path, filband):
    """Write bands.x input file."""
    content = f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{filband}'
/
"""
    with open(out_path, 'w') as f:
        f.write(content)


def write_band_labels(out_path, labels):
    """Write band_labels.dat with high-symmetry labels and their indices."""
    with open(out_path, 'w') as f:
        for label, idx in labels:
            f.write(f"{label} {idx}\n")


def main():
    n_structures = 51
    base_dir = 'calculations'
    os.makedirs(base_dir, exist_ok=True)

    for i in range(n_structures):
        nnn = f"{i:03d}"
        calc_dir = os.path.join(base_dir, nnn)
        os.makedirs(calc_dir, exist_ok=True)

        # Copy pseudopotentials
        for pp in [Ti_pseudo, O_pseudo]:
            src = os.path.join(PSEUDO_DIR, pp)
            dst = os.path.join(calc_dir, pp)
            shutil.copy2(src, dst)

        # Read the structure
        cif_path = os.path.join('structures', f'structure_{nnn}.cif')
        atoms = read(cif_path)

        # Verify nat/ntyp invariant
        nat = len(atoms)
        ntyp = len(set(atoms.get_chemical_symbols()))
        assert nat == 6, f"Structure {nnn}: expected 6 atoms, got {nat}"
        assert ntyp == 2, f"Structure {nnn}: expected 2 types, got {ntyp}"

        # Get the k-path from this specific structure
        prim, kpoints, labels, path = get_kpath(atoms)

        # Write SCF input
        pw_in_path = os.path.join(calc_dir, 'pw.in')
        write_pw_in(prim, pw_in_path, kpts=KPTS_SCF)

        # Write bands input
        bands_in_path = os.path.join(calc_dir, 'bands.in')
        write_bands_in(prim, bands_in_path, kpoints=kpoints)

        # Write bands post-processing input
        bands_post_path = os.path.join(calc_dir, 'bands_post.in')
        filband = f"{nnn}.bands.dat"
        write_bands_post_in(bands_post_path, filband)

        # Write band labels
        labels_path = os.path.join(calc_dir, 'band_labels.dat')
        write_band_labels(labels_path, labels)

        if i < 3 or i == n_structures - 1:
            print(f"Wrote {calc_dir}/: pw.in, bands.in, bands_post.in, band_labels.dat")
            print(f"  k-points: {len(kpoints)}, labels: {labels}")

    print(f"\nDone. Set up {n_structures} calculation directories in {base_dir}/")


if __name__ == '__main__':
    main()
