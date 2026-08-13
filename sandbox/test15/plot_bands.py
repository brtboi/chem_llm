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
    pattern = r"x coordinate\\s+([0-9Ee+.-]+)"

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

# PLOT
def plot_bands(calc_dir):

    calc_dir = Path(calc_dir)
    prefix = calc_dir.name
    bands_file = calc_dir / f"{prefix}.bands.dat.gnu"
    symm_file = calc_dir / "bands_post.out"
    bands = read_bands(bands_file)
    xcoords = read_symmetry_points(symm_file)

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

    # labels
    labels = ["R", r"$\\Gamma$", "X" , "M", r"$\\Gamma$"]

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
