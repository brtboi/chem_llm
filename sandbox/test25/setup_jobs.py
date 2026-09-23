import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import seekpath
from ase.io import read
from ase.calculators.espresso import Espresso
from ase.dft.kpoints import BandPath

# SETTINGS
OVERWRITE = False
BANDS_POINTS_PER_SEGMENT = 10

TEMPLATE_DIR = "template"
STRUCTURE_DIR = "structures"
CALC_DIR = "calculations"

# Pseudopotential files (pbesol, stringent, NC, SR -- see get_pseudopotential calls)
TI_PSP = "Ti.pbesol-srn-rrkjus_psl.1.0.0.UPF"
O_PSP = "O.pbesol-srn-rrkjus_psl.1.0.0.UPF"

# Energy cutoff: 42.0 Ha * 13.605693 Ry/Ha = 571.44 Ry, round up to 572.0 Ry
ECUTWFC = 572.0
ECUTRHO = 4.0 * ECUTWFC  # 2288.0 Ry, standard for norm-conserving pseudopotentials

# K-point mesh for SCF (rutile TiO2, 12 atoms, 8x8x8 is sufficient)
SCF_KPTS = (8, 8, 8)

# Wrapper script content: loads the espresso module in the same shell that
# execs the binary, and pins OMP_NUM_THREADS=1 to avoid oversubscribing the
# 128-core node with a serial pw.x/bands.x call.
PW_WRAPPER = """#!/bin/bash
module load espresso/7.5-libxc-7.0.0-cpu
export OMP_NUM_THREADS=1
exec pw.x "$@"
"""

BANDS_WRAPPER = """#!/bin/bash
module load espresso/7.5-libxc-7.0.0-cpu
export OMP_NUM_THREADS=1
exec bands.x "$@"
"""

os.makedirs(CALC_DIR, exist_ok=True)
os.makedirs(STRUCTURE_DIR, exist_ok=True)


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
    # silently under-detect symmetry.
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


def build_bandpath_object(structure, band_kpts, band_labels):
    """Construct an ASE BandPath object from the seekpath-derived path.

    The BandPath object is needed to pass to the ASE Espresso calculator's
    kpts parameter as kpts={'path': bandpath} for the bands calculation.

    The path string is built from the labels in order.  The special_points
    dict maps each label to its fractional coordinates in the actual cell's
    reciprocal lattice (already computed by compute_band_path).
    """
    # Build the path string from the labels in order
    path_str = ""
    for i, (label, _) in enumerate(band_labels):
        if i > 0:
            path_str += ""
        path_str += label

    # Build special_points dict: label -> fractional coords in actual cell
    special_points = {}
    for label, idx in band_labels:
        # The k-point at this index is the vertex for this label
        special_points[label] = band_kpts[idx, :3]

    # The kpts array for BandPath: (N, 3) fractional coordinates
    kpts_3d = band_kpts[:, :3]

    return BandPath(
        cell=structure.cell[:],
        kpts=kpts_3d,
        special_points=special_points,
        path=path_str,
    )


def write_wrapper_scripts(calc_path):
    """Write pw_wrapper.sh and bands_wrapper.sh to the calculation directory.

    These wrapper scripts load the espresso module in the same shell that
    execs the binary (required because Lmod modules are not active in a
    plain subprocess), and pin OMP_NUM_THREADS=1 to prevent a serial
    pw.x/bands.x call from oversubscribing the 128-core node.
    """
    pw_wrapper_path = calc_path / "pw_wrapper.sh"
    bands_wrapper_path = calc_path / "bands_wrapper.sh"

    with open(pw_wrapper_path, "w") as f:
        f.write(PW_WRAPPER)
    os.chmod(pw_wrapper_path, 0o755)

    with open(bands_wrapper_path, "w") as f:
        f.write(BANDS_WRAPPER)
    os.chmod(bands_wrapper_path, 0o755)


def write_submit_script(calc_path, prefix):
    """Write submit.sh for the calculation directory.

    This script runs SCF, bands pw.x, and bands.x in sequence.
    It uses the wrapper scripts (which load the module and set OMP_NUM_THREADS=1)
    and runs everything as a plain (non-MPI) process on the already-allocated
    compute node.
    """
    submit_text = f"""#!/bin/bash
#SBATCH -A m4735
#SBATCH -J tio2_{prefix}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=brent.hu@yale.edu

echo "Running SCF for {prefix}"
./pw_wrapper.sh -in pw.in > pw.out 2>&1

if [ $? -ne 0 ]; then
    echo "SCF failed"
    exit 1
fi

echo "Running bands pw.x for {prefix}"
./pw_wrapper.sh -in bands.in > bands_pw.out 2>&1

if [ $? -ne 0 ]; then
    echo "Bands calculation failed"
    exit 1
fi

echo "Running bands.x for {prefix}"
./bands_wrapper.sh -in bands_post.in > bands_post.out 2>&1

echo "Finished {prefix}"
"""
    with open(calc_path / "submit.sh", "w") as f:
        f.write(submit_text)
    os.chmod(calc_path / "submit.sh", 0o755)


def run_scf(calc_path, structure, prefix):
    """Run the SCF calculation using ASE's Espresso calculator.

    The ASE Espresso calculator writes the input file and runs pw.x via
    the wrapper script (which loads the module and sets OMP_NUM_THREADS=1).
    """
    calc = Espresso(
        command=str(calc_path / "pw_wrapper.sh"),
        directory=str(calc_path),
        label=prefix,
        # CONTROL
        prefix=prefix,
        calculation="scf",
        restart_mode="from_scratch",
        outdir="./",
        wfcdir="./",
        pseudo_dir="./",
        verbosity="high",
        # SYSTEM
        ibrav=0,
        nat=len(structure),
        ntyp=len(set(structure.get_chemical_symbols())),
        ecutwfc=ECUTWFC,
        ecutrho=ECUTRHO,
        tot_charge=0.0,
        nosym=True,
        noinv=True,
        occupations="fixed",
        # ELECTRONS
        electron_maxstep=100,
        conv_thr=1.0e-8,
        mixing_mode="plain",
        mixing_beta=0.3,
        mixing_ndim=8,
        diagonalization="david",
        diago_david_ndim=4,
        diago_full_acc=False,
        # Pseudopotentials
        pseudopotentials={"Ti": TI_PSP, "O": O_PSP},
        # K-points
        kpts=SCF_KPTS,
    )
    structure.calc = calc
    energy = structure.get_potential_energy()
    print(f"  SCF energy: {energy:.6f} eV")
    return energy


def run_bands_pw(calc_path, structure, prefix, band_kpts, band_labels):
    """Run the bands pw.x calculation using ASE's Espresso calculator.

    The bands calculation uses the explicit k-point list derived from
    seekpath.  We construct a BandPath object and pass it to the calculator.
    """
    bandpath = build_bandpath_object(structure, band_kpts, band_labels)

    calc = Espresso(
        command=str(calc_path / "pw_wrapper.sh"),
        directory=str(calc_path),
        label=prefix,
        # CONTROL
        prefix=prefix,
        calculation="bands",
        restart_mode="from_scratch",
        outdir="./",
        wfcdir="./",
        pseudo_dir="./",
        verbosity="high",
        # SYSTEM
        ibrav=0,
        nat=len(structure),
        ntyp=len(set(structure.get_chemical_symbols())),
        ecutwfc=ECUTWFC,
        ecutrho=ECUTRHO,
        tot_charge=0.0,
        nosym=True,
        noinv=True,
        occupations="fixed",
        # ELECTRONS
        electron_maxstep=100,
        conv_thr=1.0e-8,
        mixing_mode="plain",
        mixing_beta=0.3,
        mixing_ndim=8,
        diagonalization="david",
        diago_david_ndim=4,
        diago_full_acc=False,
        # Pseudopotentials
        pseudopotentials={"Ti": TI_PSP, "O": O_PSP},
        # K-points: BandPath object for the bands calculation
        kpts={"path": bandpath},
    )
    structure.calc = calc
    energy = structure.get_potential_energy()
    print(f"  Bands pw.x energy: {energy:.6f} eV")
    return energy


def run_bands_x(calc_path, prefix):
    """Run bands.x post-processing via subprocess.

    ASE has no native support for bands.x, so we call it directly via the
    wrapper script (which loads the module and sets OMP_NUM_THREADS=1).
    """
    bands_post_in = calc_path / "bands_post.in"
    bands_post_out = calc_path / "bands_post.out"

    result = subprocess.run(
        [str(calc_path / "bands_wrapper.sh"), "-in", str(bands_post_in)],
        cwd=str(calc_path),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  bands.x failed for {prefix}:")
        print(result.stderr)
        raise RuntimeError(f"bands.x failed for {prefix}")

    print(f"  bands.x completed for {prefix}")


def write_bands_post_in(calc_path, prefix):
    """Write the bands_post.in file for bands.x."""
    bands_post = f"""&BANDS
    prefix  = '{prefix}'
    outdir  = './'
    filband = '{prefix}.bands.dat'
    lsym = .true.,
    /
    """
    with open(calc_path / "bands_post.in", "w") as f:
        f.write(bands_post)


# MAIN LOOP
cif_files = sorted(Path(STRUCTURE_DIR).glob("structure_*.cif"))

for n, cif_file in enumerate(cif_files):
    prefix = f"{n:03d}"
    calc_path = Path(CALC_DIR) / prefix

    print(f"Setting up {calc_path}")

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

    # INVARIANT 1: nat/ntyp must be computed from the actual structure
    nat = len(structure)
    ntyp = len(set(structure.get_chemical_symbols()))
    print(f"  Structure {prefix}: {nat} atoms, {ntyp} species")

    # write wrapper scripts
    write_wrapper_scripts(calc_path)

    # derive this structure's own bands path (never reuse another
    # structure's path -- symmetry can break under perturbation)
    band_kpts, band_labels = compute_band_path(structure)

    # write band_labels.dat for plot_bands.py
    with open(calc_path / "band_labels.dat", "w") as f:
        for label, idx in band_labels:
            f.write(f"{label} {idx}\n")

    # write bands_post.in
    write_bands_post_in(calc_path, prefix)

    # write submit.sh
    write_submit_script(calc_path, prefix)

    # Run SCF
    print(f"  Running SCF for {prefix}...")
    run_scf(calc_path, structure, prefix)

    # Run bands pw.x
    print(f"  Running bands pw.x for {prefix}...")
    run_bands_pw(calc_path, structure, prefix, band_kpts, band_labels)

    # Run bands.x
    print(f"  Running bands.x for {prefix}...")
    run_bands_x(calc_path, prefix)

    print(f"  Done with {prefix}")

print("Done.")
