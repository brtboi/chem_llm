import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RY_TO_EV = 13.605693

# SETTINGS
N_OCC = 176

YMIN = -5
YMAX = 6

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
    vbm_ry = np.max(occupied_band[:,1])

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

_LATEX_LABELS = {"GAMMA": r"$\Gamma$", "SIGMA": r"$\Sigma$", "DELTA": r"$\Delta$"}

def label_to_latex(label):
    return _LATEX_LABELS.get(label.upper(), label)

# PLOT
def plot_bands(calc_dir):

    calc_dir = Path(calc_dir)
    prefix = calc_dir.name
    bands_file = calc_dir / f"{prefix}.bands.dat.gnu"
    symm_file = calc_dir / "bands_post.out"
    labels_file = calc_dir / "band_labels.dat"
    bands = read_bands(bands_file)
    xcoords = read_symmetry_points(symm_file)
    band_labels = read_band_labels(labels_file)

    # VBM alignment
    vbm_eV = compute_vbm(bands)

    print(f"{prefix}: VBM = {vbm_eV:.6f} eV")

    # plot
    fig, ax = plt.subplots(figsize=(6,5))
    for band in bands:

        k = band[:,0]
        e = band[:,1] - vbm_eV

        ax.plot(
            k,
            e,
            lw=1.2,
            color="blue"
        )

    # symmetry lines
    for x in xcoords:

        ax.axvline(
            x,
            lw=0.7,
            color="black",
            alpha=0.5
        )

    ax.axhline(
        0.0,
        lw=0.8,
        color="black"
    )

    # labels -- one per vertex bands.x reports, in the same order
    # compute_band_path wrote them in setup_jobs.py.
    labels = [label_to_latex(label) for label, _ in band_labels]

    ax.set_xticks(xcoords)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Energy (eV)")
    ax.set_xlim(xcoords[0], xcoords[-1])
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
