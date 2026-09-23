#!/usr/bin/env python
"""
Setup QE calculation directories for all 51 rutile TiO2 structures.

For each structure NNN (000-050), creates calculations/NNN/ containing:
  - Pseudopotential files (copied from template/)
  - pw.in: SCF input (written with ase.io.espresso.write_espresso_in)
  - bands.in: non-self-consistent bands input along a k-path
  - bands_post.in: bands.x post-processing input
  - band_labels.dat: high-symmetry labels and their indices in the k-point list
  - submit.sh: SLURM script running pw.x SCF, pw.x bands, then bands.x

K-path derivation (INVARIANT 3):
  Uses seekpath with symprec=0.01 (matching pymatgen's default, avoiding
  spglib's tight 1e-5 default which can under-detect symmetry on CIF
  round-tripped coordinates). The path is reprojected from seekpath's
  internal primitive reciprocal lattice onto the actual cell's reciprocal
  lattice via Cartesian coordinates, with the 2*pi convention handled
  explicitly. 20 points per segment are interpolated.

No pymatgen is used anywhere in this script.
"""

import os
import shutil
import numpy as np
from ase.io import read
from ase.io.espresso import write_espresso_in
import seekpath
import spglib

# --- Constants ---
STRUCTURES_DIR = "structures"
TEMPLATE_DIR = "template"
CALC_DIR = "calculations"

# Pseudopotential files (from template/)
TI_PSP = "Ti.pbesol-spn-rrkjus_psl.1.0.0.UPF"
O_PSP = "O.pbesol-spn-rrkjus_psl.1.0.0.UPF"

# Pseudopotential dict for write_espresso_in (species -> filename)
PSEUDOPOTENTIALS = {
    "Ti": TI_PSP,
    "O": O_PSP,
}

# QE settings (from STEP 4 note)
PREFIX = "pwscf"
OUTDIR = "./tmp"
ECUTWFC = 42.0  # Ry (from pseudopotential hint, both Ti and O recommend 42.0 Ha = 42.0 Ry)
ECUTRHO = 4 * ECUTWFC  # 168.0 Ry (default for norm-conserving PPs)
CONV_THR = 1.0e-8
SCF_KPTS = (4, 4, 4)  # Monkhorst-Pack grid for SCF
# INVARIANT 2: nbnd must be >= occupied bands. From pw.out: 96 electrons / 2 = 48 occupied bands.
# Use 64 for generous padding above 48.
NBND_BANDS = 64
POINTS_PER_SEGMENT = 20  # k-points per path segment

# SLURM settings
SLURM_ACCOUNT = "m4735"
SLURM_EMAIL = "brent.hu@yale.edu"


def get_k_path(atoms):
    """
    Derive the k-path for a band structure calculation using seekpath.

    INVARIANT 3: The path must be derived from the EXACT same ASE Atoms
    object that will be written into CELL_PARAMETERS. We use seekpath with
    symprec=0.01 (not spglib's tight 1e-5 default) to avoid under-detecting
    symmetry on CIF round-tripped coordinates.

    The path is reprojected from seekpath's internal primitive reciprocal
    lattice onto the actual cell's reciprocal lattice via Cartesian
    coordinates, with the 2*pi convention handled explicitly.

    Returns:
        kpts_array: (n_kpts, 4) numpy array in crystal coordinates
        labels: list of (label, index) tuples for high-symmetry points
        path_segments: list of (label_from, label_to) segments
    """
    # Build the cell tuple for seekpath/spglib
    cell_tuple = (
        atoms.cell[:],
        atoms.get_scaled_positions(),
        atoms.get_atomic_numbers(),
    )

    # Get the path from seekpath with symprec=0.01
    res = seekpath.get_path(cell_tuple, symprec=0.01)

    # INVARIANT 3: Assert space group number matches spglib's result
    spglib_res = spglib.get_symmetry_dataset(cell_tuple, symprec=0.01)
    assert res['spacegroup_number'] == spglib_res.number, (
        f"Space group mismatch: seekpath={res['spacegroup_number']}, "
        f"spglib={spglib_res.number}"
    )

    # Get the reciprocal primitive lattice from seekpath (already includes 2*pi)
    recip_prim = np.array(res['reciprocal_primitive_lattice'])

    # Get the actual cell's reciprocal lattice (ASE has NO 2*pi factor -- must add it)
    recip_actual = 2 * np.pi * np.array(atoms.cell.reciprocal()[:])

    # Function to convert fractional coordinates from seekpath's primitive
    # reciprocal lattice to the actual cell's reciprocal lattice
    def to_actual_frac(frac_prim):
        # Convert to Cartesian coordinates (in 2*pi/alat units)
        cart = np.array(frac_prim) @ recip_prim
        # Convert back to fractional coordinates in the actual cell's reciprocal lattice
        return cart @ np.linalg.inv(recip_actual)

    # Get the point coordinates in the actual cell's fractional coordinates
    point_coords_actual = {
        lbl: to_actual_frac(c) for lbl, c in res['point_coords'].items()
    }

    # Build the interpolated k-point list
    # res['path'] is a list of (label_from, label_to) segments
    path_segments = res['path']
    kpts_list = []
    labels = []  # (label, index) for high-symmetry points

    for i_seg, (lbl_from, lbl_to) in enumerate(path_segments):
        coord_from = point_coords_actual[lbl_from]
        coord_to = point_coords_actual[lbl_to]

        # Add the starting point of this segment (skip if it's the first point overall)
        if i_seg == 0:
            kpts_list.append(coord_from)
            labels.append((lbl_from, 0))

        # Interpolate points between coord_from and coord_to
        # We add POINTS_PER_SEGMENT points, excluding the starting point
        # (already added) but including the ending point
        for i in range(1, POINTS_PER_SEGMENT + 1):
            frac = i / POINTS_PER_SEGMENT
            coord = (1 - frac) * coord_from + frac * coord_to
            kpts_list.append(coord)

        # Add the label for the ending point
        labels.append((lbl_to, len(kpts_list) - 1))

    # Build the (n_kpts, 4) array: 3 fractional coords + weight (1.0, unused for bands)
    kpts_array = np.zeros((len(kpts_list), 4))
    for i, coord in enumerate(kpts_list):
        kpts_array[i, :3] = coord
        kpts_array[i, 3] = 1.0  # weight (unused for non-self-consistent bands run)

    return kpts_array, labels, path_segments


def write_pw_in(atoms, calc_dir, calculation, kpts, nbnd=None):
    """
    Write a pw.x input file using ase.io.espresso.write_espresso_in.

    Args:
        atoms: ASE Atoms object
        calc_dir: directory to write the file to
        calculation: 'scf' or 'bands'
        kpts: k-point specification (tuple for MP grid, or (n,4) array for explicit)
        nbnd: number of bands (only for bands calculation)
    """
    # Build the input_data dictionary
    input_data = {
        "CONTROL": {
            "calculation": calculation,
            "prefix": PREFIX,
            "outdir": OUTDIR,
            "pseudo_dir": "./",
            "verbosity": "high",
        },
        "SYSTEM": {
            "ecutwfc": ECUTWFC,
            "ecutrho": ECUTRHO,
            "occupations": "fixed",
        },
        "ELECTRONS": {
            "conv_thr": CONV_THR,
        },
    }

    # Add nbnd for bands calculation
    if nbnd is not None:
        input_data["SYSTEM"]["nbnd"] = nbnd

    # Write the input file
    in_path = os.path.join(calc_dir, "pw.in" if calculation == "scf" else "bands.in")
    with open(in_path, "w") as f:
        write_espresso_in(
            f,
            atoms,
            input_data=input_data,
            pseudopotentials=PSEUDOPOTENTIALS,
            kpts=kpts,
            crystal_coordinates=True,
        )

    return in_path


def write_bands_post_in(calc_dir, nnn):
    """
    Write the bands.x post-processing input file.
    """
    filband = f"{nnn}.bands.dat"
    content = f"""&BANDS
    prefix = '{PREFIX}'
    outdir = '{OUTDIR}'
    filband = '{filband}'
/
"""
    post_path = os.path.join(calc_dir, "bands_post.in")
    with open(post_path, "w") as f:
        f.write(content)
    return post_path


def write_band_labels(calc_dir, labels):
    """
    Write the band_labels.dat file with high-symmetry labels and their indices.
    """
    labels_path = os.path.join(calc_dir, "band_labels.dat")
    with open(labels_path, "w") as f:
        for label, index in labels:
            f.write(f"{label} {index}\n")
    return labels_path


def write_submit_sh(calc_dir, nnn):
    """
    Write the SLURM submit script.
    """
    content = f"""#!/bin/bash
#SBATCH --account={SLURM_ACCOUNT}
#SBATCH --mail-type=ALL
#SBATCH --mail-user={SLURM_EMAIL}
#SBATCH --job-name=TiO2_{nnn}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

module load espresso

# SCF calculation
pw.x < pw.in > pw.out

# Bands calculation
pw.x < bands.in > bands.out

# Bands post-processing
bands.x < bands_post.in > bands_post.out
"""
    sh_path = os.path.join(calc_dir, "submit.sh")
    with open(sh_path, "w") as f:
        f.write(content)
    os.chmod(sh_path, 0o755)
    return sh_path


def main():
    # Create the calculations directory
    os.makedirs(CALC_DIR, exist_ok=True)

    # Process each structure
    for nnn in range(51):
        nnn_str = f"{nnn:03d}"
        cif_path = os.path.join(STRUCTURES_DIR, f"structure_{nnn_str}.cif")
        calc_dir = os.path.join(CALC_DIR, nnn_str)

        print(f"Processing structure {nnn_str}...")

        # Create the calculation directory
        os.makedirs(calc_dir, exist_ok=True)

        # Copy pseudopotential files
        for psp in [TI_PSP, O_PSP]:
            src = os.path.join(TEMPLATE_DIR, psp)
            dst = os.path.join(calc_dir, psp)
            shutil.copy2(src, dst)

        # Read the structure
        atoms = read(cif_path, format="cif")

        # INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
        nat = len(atoms)
        ntyp = len(set(atoms.get_chemical_symbols()))
        print(f"  nat={nat}, ntyp={ntyp}")

        # Get the k-path for this structure (INVARIANT 3: per-structure path)
        kpts_array, labels, path_segments = get_k_path(atoms)
        print(f"  K-path: {len(kpts_array)} points, {len(labels)} high-symmetry labels")
        print(f"  Path segments: {path_segments}")

        # Write SCF input (pw.in)
        pw_in_path = write_pw_in(atoms, calc_dir, "scf", kpts=SCF_KPTS)
        print(f"  Wrote {pw_in_path}")

        # Write bands input (bands.in)
        bands_in_path = write_pw_in(atoms, calc_dir, "bands", kpts=kpts_array, nbnd=NBND_BANDS)
        print(f"  Wrote {bands_in_path}")

        # Write bands post-processing input (bands_post.in)
        bands_post_path = write_bands_post_in(calc_dir, nnn_str)
        print(f"  Wrote {bands_post_path}")

        # Write band labels
        labels_path = write_band_labels(calc_dir, labels)
        print(f"  Wrote {labels_path}")

        # Write submit script
        sh_path = write_submit_sh(calc_dir, nnn_str)
        print(f"  Wrote {sh_path}")

    print("\nDone: all calculation directories set up.")


if __name__ == "__main__":
    main()
