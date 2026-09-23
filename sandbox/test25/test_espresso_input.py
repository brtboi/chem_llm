import os
import shutil
import numpy as np
import seekpath
from ase.io import read
from ase.calculators.espresso import Espresso, EspressoProfile

# Read the base structure
structure = read("structures/structure_000.cif")
print(f"Structure: {len(structure)} atoms, {len(set(structure.get_chemical_symbols()))} species")

# Compute band path using seekpath
cell_tuple = (
    structure.cell[:],
    structure.get_scaled_positions(),
    structure.get_atomic_numbers(),
)
res = seekpath.get_path(cell_tuple, symprec=0.01)
print(f"Space group: {res['spacegroup_number']}")
print(f"Path: {res['path']}")

# Reproject to actual cell
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

# Build kpts list
BANDS_POINTS_PER_SEGMENT = 10
kpts = []
labels = []
for seg_from, seg_to in res["path"]:
    start = point_coords_actual[seg_from]
    end = point_coords_actual[seg_to]
    labels.append((seg_from, len(kpts)))
    for i in range(BANDS_POINTS_PER_SEGMENT):
        frac = i / BANDS_POINTS_PER_SEGMENT
        kpts.append(start * (1 - frac) + end * frac)
labels.append((res["path"][-1][1], len(kpts)))
kpts.append(point_coords_actual[res["path"][-1][1]])

kpts = np.array(kpts)
weights = np.ones((len(kpts), 1))
band_kpts = np.hstack([kpts, weights])
print(f"Band kpts shape: {band_kpts.shape}")
print(f"Labels: {labels}")

# Build path string and special_points dict for the kpts dict
path_str = "".join(label for label, _ in labels)
special_points = {label: band_kpts[idx, :3] for label, idx in labels}
print(f"Path string: {path_str}")
print(f"Special points: {special_points}")

# Create a test directory
test_dir = "test_espresso"
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
os.makedirs(test_dir, exist_ok=True)

# Write a minimal wrapper script
with open(os.path.join(test_dir, "pw_wrapper.sh"), "w") as f:
    f.write('#!/bin/bash\nmodule load espresso/7.5-libxc-7.0.0-cpu\nexport OMP_NUM_THREADS=1\nexec pw.x "$@"\n')
os.chmod(os.path.join(test_dir, "pw_wrapper.sh"), 0o755)

# Copy pseudopotentials
shutil.copy("template/Ti.pbesol-srn-rrkjus_psl.1.0.0.UPF", test_dir)
shutil.copy("template/O.pbesol-srn-rrkjus_psl.1.0.0.UPF", test_dir)

# Create the Espresso calculator for SCF using EspressoProfile
profile = EspressoProfile(
    command=os.path.join(test_dir, "pw_wrapper.sh"),
    pseudo_dir="./",
)

calc_scf = Espresso(
    profile=profile,
    directory=test_dir,
    prefix="test_scf",
    calculation="scf",
    restart_mode="from_scratch",
    outdir="./",
    wfcdir="./",
    verbosity="high",
    ibrav=0,
    nat=len(structure),
    ntyp=len(set(structure.get_chemical_symbols())),
    ecutwfc=572.0,
    ecutrho=2288.0,
    tot_charge=0.0,
    nosym=True,
    noinv=True,
    occupations="fixed",
    electron_maxstep=100,
    conv_thr=1.0e-8,
    mixing_mode="plain",
    mixing_beta=0.3,
    mixing_ndim=8,
    diagonalization="david",
    diago_david_ndim=4,
    diago_full_acc=False,
    pseudopotentials={"Ti": "Ti.pbesol-srn-rrkjus_psl.1.0.0.UPF", "O": "O.pbesol-srn-rrkjus_psl.1.0.0.UPF"},
    kpts=(8, 8, 8),
)

# Use write_inputfiles with atoms and properties to write the input file
calc_scf.write_inputfiles(structure, ["energy"])
print("\nSCF input file written")

import glob
for f in sorted(glob.glob(os.path.join(test_dir, "*.in"))):
    print(f"\n--- {f} ---")
    with open(f) as fh:
        print(fh.read())

# Now create the bands calculator with kpts as a dict with path string and special_points
profile_bands = EspressoProfile(
    command=os.path.join(test_dir, "pw_wrapper.sh"),
    pseudo_dir="./",
)

calc_bands = Espresso(
    profile=profile_bands,
    directory=test_dir,
    prefix="test_bands",
    calculation="bands",
    restart_mode="from_scratch",
    outdir="./",
    wfcdir="./",
    verbosity="high",
    ibrav=0,
    nat=len(structure),
    ntyp=len(set(structure.get_chemical_symbols())),
    ecutwfc=572.0,
    ecutrho=2288.0,
    tot_charge=0.0,
    nosym=True,
    noinv=True,
    occupations="fixed",
    electron_maxstep=100,
    conv_thr=1.0e-8,
    mixing_mode="plain",
    mixing_beta=0.3,
    mixing_ndim=8,
    diagonalization="david",
    diago_david_ndim=4,
    diago_full_acc=False,
    pseudopotentials={"Ti": "Ti.pbesol-srn-rrkjus_psl.1.0.0.UPF", "O": "O.pbesol-srn-rrkjus_psl.1.0.0.UPF"},
    kpts={"path": path_str, "special_points": special_points, "npoints": BANDS_POINTS_PER_SEGMENT},
)

calc_bands.write_inputfiles(structure, ["energy"])
print("\nBands input file written")

for f in sorted(glob.glob(os.path.join(test_dir, "*.in"))):
    print(f"\n--- {f} ---")
    with open(f) as fh:
        print(fh.read())

print("\nDone. Check the input files to see how ASE writes the K_POINTS card for a band path.")
