import os
import shutil
from pathlib import Path

import numpy as np
import seekpath
from ase.io import read

# SETTINGS
OVERWRITE = False
BANDS_POINTS_PER_SEGMENT = 10

TEMPLATE_DIR = "template"
STRUCTURE_DIR = "structures"
CALC_DIR = "calculations"

os.makedirs(CALC_DIR, exist_ok=True)
os.makedirs(STRUCTURE_DIR, exist_ok=True)

# HELPERS
ANG_TO_BOHR = 1.889726125


def compute_band_path(structure, points_per_segment=BANDS_POINTS_PER_SEGMENT):
    """Derive the high-symmetry bands path from THIS structure's actual
    symmetry (lattice + basis together) via seekpath, then reproject it
    onto the actual cell (not the primitive cell seekpath may report if
    `structure`'s cell isn't already primitive) through Cartesian
    coordinates.

    NEVER type high-symmetry fractional coordinates from a textbook
    space-group convention, and NEVER use ASE's `atoms.cell.bandpath()`
    for this: that method classifies the Bravais lattice from the cell
    vectors ALONE, so it never inspects the atomic basis and cannot detect
    symmetry lowered by the actual atomic arrangement (e.g. a randomly
    perturbed structure, exactly what this pipeline generates, is
    generically NOT exactly on the ideal high-symmetry point and can have
    a lower true space group than its lattice shape alone would suggest).
    seekpath uses spglib on the full (lattice + basis) structure, so it
    reflects the structure's real symmetry.

    Returns (kpts, labels) where kpts is an (N, 4) array of fractional
    k-points (in the ACTUAL cell's reciprocal lattice) with a constant
    weight column, and labels is a list of (label, cumulative_index) for
    every named point on the path, in order (for plot x-axis ticks).
    """
    cell_tuple = (
        structure.cell[:],
        structure.get_scaled_positions(),
        structure.get_atomic_numbers(),
    )
    # symprec=0.01 (not seekpath/spglib's tight 1e-5 default) matches
    # pymatgen's SpacegroupAnalyzer default tolerance -- CIF-round-tripped
    # coordinates carry enough numerical noise that the tight default can
    # silently under-detect symmetry (e.g. this exact rutile TiO2 cell from
    # Materials Project reads back as spglib-P1 at 1e-5, but the true,
    # MP-declared space group P4_2/mnm #136 is only recovered at 0.01+).
    res = seekpath.get_path(cell_tuple, symprec=0.01)

    # Both reciprocal lattices must use the same (2*pi) convention before
    # any Cartesian round-trip: seekpath's reciprocal_primitive_lattice
    # already includes the 2*pi factor, but ASE's Cell.reciprocal() does
    # NOT -- so it must be added explicitly here.
    recip_prim = np.array(res["reciprocal_primitive_lattice"])
    recip_actual = 2 * np.pi * np.array(structure.cell.reciprocal()[:])
    recip_actual_inv = np.linalg.inv(recip_actual)

    def prim_frac_to_actual_frac(frac_prim):
        cart = np.array(frac_prim) @ recip_prim
        return cart @ recip_actual_inv

    point_coords_actual = {
        label: prim_frac_to_actual_frac(coords)
        for label, coords in res["point_coords"].items()
    }

    kpts = []
    labels = []
    for seg_from, seg_to in res["path"]:
        start = point_coords_actual[seg_from]
        end = point_coords_actual[seg_to]
        labels.append((seg_from, len(kpts)))
        for i in range(points_per_segment):
            frac = i / points_per_segment
            kpts.append(start * (1 - frac) + end * frac)
    # Final point of the last segment.
    labels.append((res["path"][-1][1], len(kpts)))
    kpts.append(point_coords_actual[res["path"][-1][1]])

    kpts = np.array(kpts)
    weights = np.ones((len(kpts), 1))
    return np.hstack([kpts, weights]), labels


def structure_to_qe(structure, prefix, calculation, band_kpts=None):

    lines = []

    # CONTROL
    lines.append("&CONTROL")
    lines.append(f"   prefix = '{prefix}'")
    lines.append(f"   calculation = '{calculation}'")
    lines.append("   restart_mode = 'from_scratch'")
    lines.append("   outdir = './'")
    lines.append("   wfcdir = './'")
    lines.append("   pseudo_dir = './'")
    lines.append("   verbosity = 'high'")
    lines.append("/")

    # SYSTEM
    lines.append("&SYSTEM")
    lines.append("   ibrav = 0")
    # nat/ntyp MUST be computed from the `structure` object actually being
    # written below, never hardcoded -- a fixed number silently drifts out
    # of sync with ATOMIC_POSITIONS/ATOMIC_SPECIES the moment the structure
    # changes (different compound, supercell, or perturbed copy).
    lines.append(f"   nat = {len(structure)}")
    lines.append(f"   ntyp = {len(set(structure.get_chemical_symbols()))}")
    lines.append("   ecutwfc = 50.0")
    lines.append("   ecutrho = 250.0")
    lines.append("   tot_charge = 0.0")
    lines.append("   nosym = .true.")
    lines.append("   noinv = .true.")
    lines.append("   occupations = 'fixed'")
    lines.append("   nspin = 4")
    lines.append("   noncolin = .true.")
    lines.append("   lspinorb = .true.")

    if calculation == "bands":
        lines.append("   nbnd = 200")

    lines.append("/")

    # ELECTRONS
    lines.append("&electrons")
    lines.append("   electron_maxstep = 100")
    lines.append("   conv_thr = 1.0d-8")
    lines.append("   mixing_mode = 'plain'")
    lines.append("   mixing_beta = 0.3")
    lines.append("   mixing_ndim = 8")
    lines.append("   diagonalization = 'david'")
    lines.append("   diago_david_ndim = 4")
    lines.append("   diago_full_acc = .false.")
    lines.append("/")

    # SPECIES
    lines.append("ATOMIC_SPECIES")
    lines.append("Cs 132.90545 Cs.rel-pbe-spn-rrkjus_psl.1.0.0.UPF")
    lines.append("Pb 207.20000 Pb.rel-pbe-dn-rrkjus_psl.1.0.0.UPF")
    lines.append("Br 79.90400 Br.USPP.FR.PBE.3.4.UPF")
    lines.append("")

    # CELL PARAMETERS
    lines.append("CELL_PARAMETERS angstrom")

    for vec in structure.cell[:]:

        lines.append(
            f"{vec[0]:.10f} "
            f"{vec[1]:.10f} "
            f"{vec[2]:.10f}"
        )

    lines.append("")

    # POSITIONS
    lines.append("ATOMIC_POSITIONS crystal")

    frac = structure.get_scaled_positions() % 1.0

    for symbol, pos in zip(structure.get_chemical_symbols(), frac):

        lines.append(
            f"{symbol:<2} "
            f"{pos[0]:.6f} "
            f"{pos[1]:.6f} "
            f"{pos[2]:.6f}"
        )

    lines.append("")

    # KPOINTS
    if calculation == "scf":

        lines.append("K_POINTS automatic")
        lines.append("8 8 8 0 0 0")

    elif calculation == "bands":

        # Explicit, pre-interpolated k-point list (one line per point, a
        # constant placeholder weight) under the plain "crystal" card --
        # physically identical to "crystal_b" (which asks pw.x to do the
        # interpolation itself from a few vertices): the interpolation
        # already happened in compute_band_path, in the ACTUAL cell's
        # fractional coordinates, from a symmetry analysis of the actual
        # structure (see compute_band_path's docstring for why that
        # matters).
        lines.append("K_POINTS crystal")
        lines.append(str(len(band_kpts)))
        for kx, ky, kz, w in band_kpts:
            lines.append(f"{kx:.10f} {ky:.10f} {kz:.10f} {w:.6f}")

    return "\n".join(lines)

def write_submit_script(calc_path, prefix):

    submit_text = f"""#!/bin/bash
#SBATCH -A m4868
#SBATCH -J cs_{prefix}
#SBATCH -C gpu
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time 03:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daniel_chabeda@berkeley.edu

module load espresso

PW=pw.x
BANDS=bands.x

echo "Running SCF for {prefix}"

srun -n 8 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 $PW -in pw.in > pw.out

if [ $? -ne 0 ]; then
    echo "SCF failed"
    exit 1
fi

echo "Running bands SCF for {prefix}"

srun -n 8 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 $PW -in bands.in > bands_pw.out

if [ $? -ne 0 ]; then
    echo "Bands calculation failed"
    exit 1
fi

echo "Running bands.x for {prefix}"

$BANDS -in bands_post.in > bands_post.out

echo "Finished {prefix}"
"""

    with open(calc_path / "submit.sh", "w") as f:
        f.write(submit_text)

# MAIN LOOP
cif_files = sorted(Path(STRUCTURE_DIR).glob("*.cif"))

for n, cif_file in enumerate(cif_files):

    prefix = f"{n:03d}"

    calc_path = Path(CALC_DIR) / prefix

    print("Setting up", calc_path)

    # copy template
    if calc_path.exists():
        if OVERWRITE:
            print(f"Overwriting {calc_path}")
            shutil.rmtree(calc_path)
        else:
            print(f"Skipping existing directory: {calc_path}")
            continue

    shutil.copytree(TEMPLATE_DIR, calc_path)

    # load structure
    structure = read(cif_file)

    # write pw.in
    pw_text = structure_to_qe(
        structure,
        prefix,
        calculation="scf"
    )

    with open(calc_path / "pw.in", "w") as f:
        f.write(pw_text)

    # derive this structure's own bands path (never reuse another
    # structure's path -- symmetry can break under perturbation)
    band_kpts, band_labels = compute_band_path(structure)

    # write bands.in
    bands_text = structure_to_qe(
        structure,
        prefix,
        calculation="bands",
        band_kpts=band_kpts,
    )

    with open(calc_path / "bands.in", "w") as f:
        f.write(bands_text)

    # record the labels/positions plot_bands.py needs for x-axis ticks, so
    # it never has to re-derive (or worse, re-hardcode) the path itself.
    with open(calc_path / "band_labels.dat", "w") as f:
        for label, idx in band_labels:
            f.write(f"{label} {idx}\n")

    # write bands_post.in
    bands_post = f"""&BANDS
    prefix  = '{prefix}'
    outdir  = './'
    filband = '{prefix}.bands.dat'
    lsym = .true.,
    /
    """

    with open(calc_path / "bands_post.in", "w") as f:
        f.write(bands_post)

    # write submit.sh
    write_submit_script(calc_path, prefix)


print("Done.")
