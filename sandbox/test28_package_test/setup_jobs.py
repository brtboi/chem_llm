#!/usr/bin/env python3
"""
Setup QE calculation directories for all 51 TiO2 structures.

For each structure NNN, creates calculations/NNN/ containing:
- Pseudopotential files (copied from template/)
- pw.in (SCF input, written with write_espresso_in)
- bands.in (non-self-consistent bands input, written with write_espresso_in)
- bands_post.in (bands.x post-processing input)
- band_labels.dat (high-symmetry labels and their indices in the k-point list)

The k-path is derived from seekpath on the SAME Atoms object written into
CELL_PARAMETERS, per INVARIANT 3. The primitive cell from seekpath is used
directly, so the k-path coordinates apply unchanged.
"""

import os
import shutil
import numpy as np
import seekpath
import spglib
from ase.io import read, write
from ase.io.espresso import write_espresso_in, write_fortran_namelist
from ase import Atoms

# --- Constants ---
ECUTWFC_RY = 84.0   # 42.0 Ha * 2 (INVARIANT 5: 1 Ha = 2 Ry)
ECUTRHO_RY = 336.0  # 4 * ecutwfc
CONV_THR = 1.0e-8   # in &ELECTRONS (INVARIANT 6)
NBND = 40           # padded generously for semicore pseudopotentials
KPTS_PER_SEGMENT = 20

# Pseudopotential files (from template/)
PP_FILES = {
    'Ti': 'template/Ti.pbesol-spn-rrkjus_psl.1.0.0.UPF',
    'O': 'template/O.pbesol-spn-rrkjus_psl.1.0.0.UPF',
}


def build_kpath(atoms):
    """
    Derive the k-path from the EXACT same Atoms object using seekpath.
    Returns:
      - prim: the primitive cell Atoms object (to write into CELL_PARAMETERS)
      - kpoints: (n_kpts, 4) array of crystal coordinates with weights
      - labels: list of (label, index) for each high-symmetry point
      - path_segments: list of (label_from, label_to) for plotting
    """
    cell_tuple = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    res = seekpath.get_path(cell_tuple, symprec=0.01)

    # Verify space group consistency (INVARIANT 3)
    dataset = spglib.get_symmetry_dataset(cell_tuple, symprec=0.01)
    assert res['spacegroup_number'] == dataset.number, \
        f"Space group mismatch: seekpath={res['spacegroup_number']}, spglib={dataset.number}"

    # Build the primitive cell from seekpath's own reduced cell
    prim = Atoms(
        cell=res['primitive_lattice'],
        scaled_positions=res['primitive_positions'],
        numbers=res['primitive_types'],
        pbc=True
    )

    # Build the explicit k-point list from the path segments
    # Each segment contributes its own start point + interior points
    # (INVARIANT 3: path is frequently discontinuous)
    kpoints = []
    labels = []  # (label, index_in_kpoints_list)
    path_segments = []

    for i_seg, (label_from, label_to) in enumerate(res['path']):
        coord_from = np.array(res['point_coords'][label_from])
        coord_to = np.array(res['point_coords'][label_to])

        # Record the start point of this segment
        if i_seg == 0:
            kpoints.append(np.append(coord_from, 1.0))
            labels.append((label_from, len(kpoints) - 1))
        else:
            # Check if this is a discontinuous jump
            prev_end = np.array(res['point_coords'][res['path'][i_seg - 1][1]])
            if not np.allclose(coord_from, prev_end, atol=1e-10):
                # Discontinuous: record both labels at the jump
                kpoints.append(np.append(coord_from, 1.0))
                labels.append((f"{res['path'][i_seg-1][1]}|{label_from}", len(kpoints) - 1))
            else:
                # Continuous: the start point is already the end of the previous segment
                # Just record the label at the existing index
                labels.append((label_from, len(kpoints) - 1))

        # Interior points (excluding the start, including the end)
        for i in range(1, KPTS_PER_SEGMENT + 1):
            t = i / KPTS_PER_SEGMENT
            coord = (1 - t) * coord_from + t * coord_to
            kpoints.append(np.append(coord, 1.0))

        # Record the end point label
        if i_seg == len(res['path']) - 1:
            labels.append((label_to, len(kpoints) - 1))
        else:
            # The end of this segment is the start of the next
            # It will be recorded when the next segment starts
            pass

        path_segments.append((label_from, label_to))

    kpoints = np.array(kpoints)
    assert kpoints.shape[1] == 4, f"kpoints shape {kpoints.shape} should be (n, 4)"

    # INVARIANT 3: assert every named point's fractional coordinates are simple fractions
    # Simple fractions include negative values (e.g. -0.5 is equivalent to 0.5 mod 1)
    for label, coord in res['point_coords'].items():
        for c in coord:
            # Check if c mod 1 is a simple fraction (0, 0.25, 0.5, 0.75)
            c_mod = c % 1.0
            simple = any(abs(c_mod - s) < 1e-10 for s in [0.0, 0.25, 0.5, 0.75])
            assert simple, f"Point {label} has non-simple coordinate {c}"

    return prim, kpoints, labels, path_segments


def write_bands_post_in(path, prefix, outdir, filband):
    """Write the bands.x post-processing input file."""
    with open(path, 'w') as f:
        write_fortran_namelist(
            f,
            input_data={
                'prefix': prefix,
                'outdir': outdir,
                'filband': filband,
            },
            binary='bands'
        )


def main():
    for nnn in range(51):
        nnn_str = f"{nnn:03d}"
        calc_dir = f"calculations/{nnn_str}"
        os.makedirs(calc_dir, exist_ok=True)

        # Read the structure
        atoms = read(f"structures/structure_{nnn_str}.cif")

        # Build the k-path from the EXACT same Atoms object (INVARIANT 3)
        prim, kpoints, labels, path_segments = build_kpath(atoms)

        # INVARIANT 1: nat and ntyp from the actual Atoms object
        nat = len(prim)
        ntyp = len(set(prim.get_chemical_symbols()))
        assert nat == 6, f"Structure {nnn_str}: expected 6 atoms, got {nat}"
        assert ntyp == 2, f"Structure {nnn_str}: expected 2 types, got {ntyp}"

        # Copy pseudopotentials into the calculation directory
        for elem, pp_path in PP_FILES.items():
            shutil.copy2(pp_path, calc_dir)

        # Pseudopotential filenames for the input file
        pseudo_dict = {
            'Ti': os.path.basename(PP_FILES['Ti']),
            'O': os.path.basename(PP_FILES['O']),
        }

        prefix = f"tio2_{nnn_str}"
        outdir = "./tmp"
        os.makedirs(os.path.join(calc_dir, "tmp"), exist_ok=True)

        # --- Write pw.in (SCF) ---
        scf_input_data = {
            'calculation': 'scf',
            'prefix': prefix,
            'outdir': outdir,
            'pseudo_dir': './',
            'ecutwfc': ECUTWFC_RY,
            'ecutrho': ECUTRHO_RY,
            'occupations': 'fixed',
            'conv_thr': CONV_THR,
        }

        with open(os.path.join(calc_dir, 'pw.in'), 'w') as f:
            write_espresso_in(
                f,
                prim,
                input_data=scf_input_data,
                pseudopotentials=pseudo_dict,
                kpts=(4, 4, 2),  # Monkhorst-Pack grid
                crystal_coordinates=True,
            )

        # --- Write bands.in (non-self-consistent bands) ---
        bands_input_data = {
            'calculation': 'bands',
            'prefix': prefix,
            'outdir': outdir,
            'pseudo_dir': './',
            'ecutwfc': ECUTWFC_RY,
            'ecutrho': ECUTRHO_RY,
            'occupations': 'fixed',
            'nbnd': NBND,
        }

        with open(os.path.join(calc_dir, 'bands.in'), 'w') as f:
            write_espresso_in(
                f,
                prim,
                input_data=bands_input_data,
                pseudopotentials=pseudo_dict,
                kpts=kpoints,  # explicit (n_kpts, 4) array in crystal coordinates
                crystal_coordinates=True,
            )

        # --- Write bands_post.in (bands.x) ---
        filband = f"{nnn_str}.bands.dat"
        write_bands_post_in(
            os.path.join(calc_dir, 'bands_post.in'),
            prefix=prefix,
            outdir=outdir,
            filband=filband,
        )

        # --- Write band_labels.dat ---
        with open(os.path.join(calc_dir, 'band_labels.dat'), 'w') as f:
            for label, idx in labels:
                f.write(f"{label}\t{idx}\n")

        if nnn == 0 or nnn % 10 == 0:
            print(f"Setup {calc_dir}: {nat} atoms, {ntyp} types, {len(kpoints)} k-points, "
                  f"{len(labels)} labels")

    print("\nDone. All 51 calculation directories set up.")


if __name__ == '__main__':
    main()
