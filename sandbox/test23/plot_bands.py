import re
import numpy as np
import matplotlib.pyplot as plt
import seekpath
from ase.io import read as ase_read
from pathlib import Path

RY_TO_EV = 13.605693

# SETTINGS
# N_OCC = number of occupied Kohn-Sham bands, read from pw.out:
# "number of Kohn-Sham states = 48" (96 electrons / 2 for spin-unpolarized).
# NEVER hardcode from a 'typical' valence-electron count per element -- the
# true count is a property of the specific pseudopotential used (semicore
# pseudopotentials include extra core-like shells as valence).
N_OCC = 48

# Energy window for the plot (eV, relative to VBM).
# Rutile TiO2 has a band gap of ~3 eV (PBEsol underestimates to ~2 eV).
# The valence bands (O 2p) extend down to ~-8 eV below VBM, and the
# conduction bands (Ti 3d) start at ~2-3 eV above VBM.
YMIN = -10
YMAX = 8

# EXTRACT HIGH-SYMMETRY POINTS
def read_symmetry_points(filename):
    xcoords = []
    pattern = r"x coordinate\s+([0-9Ee+.-]+)"

    with open(filename) as f:
        for line in f:
            if "high-symmetry point" in line:
                match = re.search(pattern, line)
                if match:
                    xcoords.append(float(match.group(1)))

    return xcoords

# READ BAND DATA
def read_bands(filename):
    bands = []
    current_band = []

    with open(filename) as f:
        for line in f:
            stripped = line.strip()

            # blank line separates bands
            if not stripped:
                if current_band:
                    bands.append(np.array(current_band))
                    current_band = []
                continue

            vals = stripped.split()

            k = float(vals[0])
            e = float(vals[1])

            current_band.append([k, e])

    if current_band:
        bands.append(np.array(current_band))

    return bands

# COMPUTE VBM
def compute_vbm(bands):
    occupied_band = bands[N_OCC - 1]
    vbm_ry = np.max(occupied_band[:, 1])

    return vbm_ry

# HIGH-SYMMETRY LABELS
# setup_jobs.py derives the bands path per-structure (via seekpath, from
# that structure's own symmetry) and records the labels it used alongside
# their index into the explicit k-point list it wrote -- read those back
# instead of hardcoding a path/label sequence here, which would silently
# mismatch any structure with different symmetry (see setup_jobs.py's
# compute_band_path docstring).
def read_band_labels(filename):
    labels = []
    with open(filename) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                labels.append((parts[0], int(parts[1])))
    return labels

_LATEX_LABELS = {
    "GAMMA": r"$\Gamma$",
    "SIGMA": r"$\Sigma$",
    "DELTA": r"$\Delta$",
    "X": r"$X$",
    "M": r"$M$",
    "A": r"$A$",
    "Z": r"$Z$",
    "R": r"$R$",
    "L": r"$L$",
    "W": r"$W$",
    "U": r"$U$",
    "K": r"$K$",
    "P": r"$P$",
    "T": r"$T$",
    "H": r"$H$",
    "N": r"$N$",
    "Q": r"$Q$",
    "S": r"$S$",
    "Y": r"$Y$",
    "B": r"$B$",
    "C": r"$C$",
    "D": r"$D$",
    "E": r"$E$",
    "F": r"$F$",
    "G": r"$G$",
    "I": r"$I$",
    "J": r"$J$",
    "O": r"$O$",
    "V": r"$V$",
}

def label_to_latex(label):
    return "|".join(_LATEX_LABELS.get(part.upper(), part) for part in label.split("|"))

# band_labels.dat only records each segment's START label, so a break in the
# path (previous segment ends at Z, next starts at X) is invisible there.
# Recompute the same seekpath path setup_jobs.py used (same structure, same
# symprec) to recover each segment's end label.
def path_segments(calc_dir):
    atoms = ase_read(calc_dir / "bands.in", format="espresso-in")
    cell = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    return seekpath.get_path(cell, symprec=0.01)["path"]

def tick_labels(band_labels, segments):
    recorded = [label for label, _ in band_labels]
    expected = [seg[0] for seg in segments] + [segments[-1][1]]
    assert recorded == expected, f"band_labels.dat {recorded} != seekpath path {expected}"
    labels = [segments[0][0]]
    for prev, seg in zip(segments, segments[1:]):
        labels.append(seg[0] if prev[1] == seg[0] else f"{prev[1]}|{seg[0]}")
    labels.append(segments[-1][1])
    return labels

# PLOT
def plot_bands(calc_dir):
    calc_dir = Path(calc_dir)
    prefix = calc_dir.name
    bands_file = calc_dir / f"{prefix}.bands.dat.gnu"
    symm_file = calc_dir / "bands_post.out"
    labels_file = calc_dir / "band_labels.dat"

    # Check that the required files exist before attempting to plot
    if not bands_file.exists():
        print(f"Skipping {prefix}: {bands_file} not found")
        return
    if not symm_file.exists():
        print(f"Skipping {prefix}: {symm_file} not found")
        return
    if not labels_file.exists():
        print(f"Skipping {prefix}: {labels_file} not found")
        return

    bands = read_bands(bands_file)
    band_labels = read_band_labels(labels_file)

    # VBM alignment
    vbm_eV = compute_vbm(bands)

    print(f"{prefix}: VBM = {vbm_eV:.6f} eV")

    # The .gnu x-column is cumulative k-path distance (2pi/alat), not a
    # point index -- map each label's index into that same column so ticks
    # and curves share units.
    k_all = bands[0][:, 0]
    tick_positions = [k_all[idx] for _, idx in band_labels]
    labels = [label_to_latex(lbl) for lbl in tick_labels(band_labels, path_segments(calc_dir))]
    assert np.isclose(tick_positions[-1], k_all.max()), "ticks and band data in different units"

    # bands.x zeroes the distance increment across a discontinuous jump in
    # the path, so a break shows up as a repeated x value; split there so no
    # vertical connector is drawn between unrelated k-points.
    breaks = [i for i in range(1, len(k_all)) if np.isclose(k_all[i], k_all[i - 1])]

    # plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for band in bands:
        for piece in np.split(band, breaks):
            ax.plot(piece[:, 0], piece[:, 1] - vbm_eV, lw=1.2, color="blue")

    # symmetry lines at the path vertices (from band_labels.dat indices)
    for x in tick_positions:
        ax.axvline(x, lw=0.7, color="black", alpha=0.5)

    ax.axhline(0.0, lw=0.8, color="black")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Energy (eV)")
    ax.set_xlim(tick_positions[0], tick_positions[-1])
    ax.set_ylim(YMIN, YMAX)

    plt.tight_layout()

    outfile = calc_dir / f"bands_{prefix}.pdf"

    plt.savefig(outfile)
    plt.close()

    print("Wrote", outfile)

# MAIN
if __name__ == "__main__":
    for calc_dir in sorted(Path("calculations").glob("*")):
        try:
            plot_bands(calc_dir)
        except Exception as e:
            print(f"Failed for {calc_dir}: {e}")
