import re
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RY_TO_EV = 13.605693

# SETTINGS
# For the 12-atom Ti4O8 conventional cell of rutile TiO2:
# Total valence electrons = 4*4 (Ti) + 8*6 (O) = 64
# N_OCC = 64 / 2 = 32 occupied bands
N_OCC = 48

# TiO2 has a band gap of ~3 eV (indirect). Set YMIN/YMAX to cover
# the VBM (at 0) through the CBM and several unoccupied bands.
YMIN = -2
YMAX = 8

# Use paths relative to this script's directory so it works regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALC_DIR = os.path.join(SCRIPT_DIR, "calculations")


# EXTRACT HIGH-SYMMETRY POINTS
def read_symmetry_points(filename):
    """Parse the x-coordinates of high-symmetry points from bands_post.out.
    
    bands.x writes lines like:
      high-symmetry point 1: x coordinate 0.000000
    We extract the numeric x coordinate for each such line.
    """
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
    """Read band data from a .bands.dat.gnu file.
    
    The .gnu format has one band per block, separated by blank lines.
    Each line within a block is: k_point energy_in_Ry
    """
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
            if len(vals) < 2:
                continue

            k = float(vals[0])
            e = float(vals[1])

            current_band.append([k, e])

    if current_band:
        bands.append(np.array(current_band))

    return bands


# COMPUTE VBM
def compute_vbm(bands):
    """Compute the valence band maximum (VBM) in eV.
    
    The VBM is the maximum energy of the highest occupied band (index N_OCC-1).
    """
    occupied_band = bands[N_OCC - 1]
    vbm_ry = np.max(occupied_band[:, 1])
    return vbm_ry * RY_TO_EV


# PLOT
def plot_bands(calc_dir):
    """Plot the band structure for a single calculation directory."""
    calc_dir = Path(calc_dir)
    prefix = calc_dir.name
    bands_file = calc_dir / f"{prefix}.bands.dat.gnu"
    symm_file = calc_dir / "bands_post.out"

    bands = read_bands(bands_file)
    xcoords = read_symmetry_points(symm_file)

    # VBM alignment: shift all energies so VBM = 0
    vbm_eV = compute_vbm(bands)
    print(f"{prefix}: VBM = {vbm_eV:.6f} eV")

    # plot
    fig, ax = plt.subplots(figsize=(6, 5))
    for band in bands:
        k = band[:, 0]
        e = band[:, 1] * RY_TO_EV - vbm_eV
        ax.plot(k, e, lw=1.2, color="blue")

    # symmetry lines
    for x in xcoords:
        ax.axvline(x, lw=0.7, color="black", alpha=0.5)

    ax.axhline(0.0, lw=0.8, color="black")

    # High-symmetry labels for tetragonal rutile TiO2:
    # Gamma(0,0,0) -> X(0.5,0,0) -> M(0.5,0.5,0) -> R(0.5,0.5,0.5) -> Gamma(0,0,0)
    labels = [r"$\Gamma$", "X", "M", "R", r"$\Gamma$"]

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
    for calc_dir in sorted(Path(CALC_DIR).glob("*")):
        try:
            plot_bands(calc_dir)
        except Exception as e:
            print(f"Failed for {calc_dir}: {e}")
