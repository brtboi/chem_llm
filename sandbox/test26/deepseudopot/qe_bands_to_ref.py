import numpy as np
import sys


def generate_qe_kpath():
    """
    Generate k-points in fractional (crystal_b) coordinates
    following the same interpolation Quantum ESPRESSO uses
    for a K_POINTS crystal_b block.

    Hardcoded to your path.
    """
    # Define your K_POINTS crystal_b block
    k_segments = [
        ([0.5, 0.5, 0.5], 10),
        ([0.0, 0.0, 0.0], 10),
        ([0.5, 0.0, 0.0], 10),
        ([0.5, 0.5, 0.0], 10),
        ([0.0, 0.0, 0.0], 1)
    ]

    # Build k-points
    kpoints = []
    for i in range(len(k_segments) - 1):
        start, npts = k_segments[i]
        end, _ = k_segments[i + 1]
        start = np.array(start)
        end = np.array(end)
        # QE includes both endpoints for each segment
        for t in range(npts):
            frac = t / npts if npts > 1 else 0.0
            k = (1 - frac) * start + frac * end
            kpoints.append(k)
    kpoints.append(k_segments[-1][0])
    return np.array(kpoints)

def parse_band_file(filename):
    """
    Parse a QE band structure file of the form:
      kx ky kz
      e1 e2 e3 ...
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    # Remove header lines (&plot ... /)
    data_lines = [ln.strip() for ln in lines if not ln.strip().startswith("&") and not ln.strip().startswith("/") and ln.strip()]

    kpoints = []
    bands = []
    i = 0
    while i < len(data_lines):
        parts = data_lines[i].split()
        if len(parts) == 3:  # k-point line
            kpt = list(map(float, parts))
            kpoints.append(kpt)
            i += 1

            # Collect band energies until next k-point or EOF
            energy_vals = []
            while i < len(data_lines):
                next_parts = data_lines[i].split()
                if len(next_parts) == 3:
                    break
                energy_vals.extend(map(float, next_parts))
                i += 1
            bands.append(energy_vals)
        else:
            i += 1

    return np.array(kpoints), np.array(bands)


def cartesian_to_fractional(k_cart, a, b, c):
    """
    Convert Cartesian k-points (in units of 1/A or 2/A)
    into fractional coordinates (crystal_b convention)
    for an orthorhombic lattice with lattice parameters a,b,c.
    """
    # Reciprocal lattice vectors in 1/A (no 2pi factor, since QE uses 2pi/a convention)
    # Fractional = Cartesian * (a / 2pi, b / 2pi, c / 2pi)
    scale = np.diag([(2 * np.pi) / a, (2 * np.pi) / b, (2 * np.pi) / c])
    #scale = np.diag([1/a, 1/b, 1/c])
    k_frac = k_cart @ scale
    return k_frac


def compute_kpath_distances(kpoints):
    """Compute cumulative distance along the k-path."""
    diffs = np.diff(kpoints, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    distances = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    return distances


def write_kpoints_file(filename, kpoints):
    """Write fractional k-points to kpoints_0.par."""
    with open(filename, "w") as f:
        for k in kpoints:
            f.write(f"{k[0]:.6f} {k[1]:.6f} {k[2]:.6f} 1.0\n")


def write_band_file(filename, distances, bands):
    """Write expBandStruct_0.par file."""
    with open(filename, "w") as f:
        for d, bvals in zip(distances, bands):
            f.write(f"{d:.6f} " + " ".join(f"{x:.6f}" for x in bvals) + "\n")


def main():
    if len(sys.argv) != 2:
        print("Usage: python qe_bands_to_ref.py <bands.dat>")
        sys.exit(1)

    input_file = sys.argv[1]
    # celldm_1, celldm_2, celldm_3 = map(float, sys.argv[2:5])
    # a = celldm_1 * 0.529177
    # b = a * celldm_2
    # c = a * celldm_3
    

    print(f"Reading {input_file}")

    k_cart, bands = parse_band_file(input_file)
    bands = bands[:, 72:]
    
    # Convert to fractional coordinates
    #k_frac = cartesian_to_fractional(k_cart, a, b, c)
    k_frac = generate_qe_kpath()

    # Compute distances along the k-path (using fractional coordinates)
    distances = compute_kpath_distances(k_frac)

    # Write outputs
    write_kpoints_file("kpoints_0.par", k_frac)
    write_band_file("expBandStruct_0.par", distances, bands)

    print(f"Wrote kpoints_0.par (fractional coords) and expBandStruct_0.par")


if __name__ == "__main__":
    main()
