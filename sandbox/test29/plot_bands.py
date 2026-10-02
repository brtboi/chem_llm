import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# plot_bands.py -- Plot the band structure for calculations/000
# =============================================================================
# Reads:
#   - calculations/000/000.bands.dat.gnu  (bands.x output: x in 2*pi/alat, E in eV)
#   - calculations/000/band_labels.dat    (high-symmetry labels and indices)
#   - calculations/000/pw.out             (number of electrons for VBM alignment)
#   - calculations/000/bands_post.out     (high-symmetry x-coordinates in 2*pi/alat)
# Writes:
#   - calculations/000/bands_000.pdf
# =============================================================================

CALC_DIR = 'calculations/000'
GNU_FILE = f'{CALC_DIR}/000.bands.dat.gnu'
LABELS_FILE = f'{CALC_DIR}/band_labels.dat'
PW_OUT = f'{CALC_DIR}/pw.out'
BANDS_POST_OUT = f'{CALC_DIR}/bands_post.out'
OUTPUT_PDF = f'{CALC_DIR}/bands_000.pdf'


def read_electron_count(pw_out_path):
    """Read 'number of electrons' from pw.out."""
    with open(pw_out_path) as f:
        for line in f:
            m = re.search(r'number of electrons\s+=\s+([\d.]+)', line)
            if m:
                return float(m.group(1))
    raise ValueError(f"Could not find 'number of electrons' in {pw_out_path}")


def read_high_symmetry_points(bands_post_out_path):
    """
    Read high-symmetry point x-coordinates from bands_post.out.
    These are in 2*pi/alat units, same as the .gnu file's x column.
    Returns list of (x_coord, label) tuples.
    """
    points = []
    with open(bands_post_out_path) as f:
        for line in f:
            m = re.search(r'high-symmetry point:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+x coordinate\s+([\d.]+)', line)
            if m:
                x_coord = float(m.group(4))
                # Determine label from fractional coordinates
                fx, fy, fz = float(m.group(1)), float(m.group(2)), float(m.group(3))
                # Map to standard labels for tetragonal rutile
                if abs(fx) < 1e-6 and abs(fy) < 1e-6 and abs(fz) < 1e-6:
                    label = 'GAMMA'
                elif abs(fx) < 1e-6 and abs(fy - 0.5) < 1e-6 and abs(fz) < 1e-6:
                    label = 'X'
                elif abs(fx - 0.5) < 1e-6 and abs(fy - 0.5) < 1e-6 and abs(fz) < 1e-6:
                    label = 'M'
                elif abs(fx) < 1e-6 and abs(fy) < 1e-6 and abs(fz - 0.7772) < 1e-3:
                    label = 'Z'
                elif abs(fx) < 1e-6 and abs(fy - 0.5) < 1e-6 and abs(fz - 0.7772) < 1e-3:
                    label = 'R'
                elif abs(fx - 0.5) < 1e-6 and abs(fy - 0.5) < 1e-6 and abs(fz - 0.7772) < 1e-3:
                    label = 'A'
                else:
                    label = f'({fx:.3f},{fy:.3f},{fz:.3f})'
                points.append((x_coord, label))
    return points


def read_band_labels(labels_file):
    """Read band_labels.dat: lines of 'index label'."""
    labels = []
    with open(labels_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                idx = int(parts[0])
                label = parts[1]
                labels.append((idx, label))
    return labels


def main():
    # 1. Read electron count from pw.out
    n_electrons = read_electron_count(PW_OUT)
    print(f"Number of electrons from pw.out: {n_electrons}")
    
    # INVARIANT 2: occupied bands = n_electrons / 2 for spin-unpolarized
    n_occ = int(n_electrons / 2)
    print(f"Occupied bands: {n_occ}")
    
    # 2. Read high-symmetry points from bands_post.out (x in 2*pi/alat)
    hs_points = read_high_symmetry_points(BANDS_POST_OUT)
    print(f"High-symmetry points from bands_post.out: {len(hs_points)}")
    for x, label in hs_points:
        print(f"  x={x:.4f}  {label}")
    
    # 3. Read band labels from band_labels.dat (indices into k-point list)
    band_labels = read_band_labels(LABELS_FILE)
    print(f"Band labels from band_labels.dat: {len(band_labels)}")
    
    # 4. Read the .gnu file: x in 2*pi/alat, E in eV
    # Format: each line is "x energy" for one band, bands are sequential
    data = np.loadtxt(GNU_FILE)
    x_col = data[:, 0]
    e_col = data[:, 1]
    
    # The .gnu file has one line per (k-point, band) pair.
    # Total lines = n_kpts * n_bands
    n_lines = len(x_col)
    
    # Determine n_kpts and n_bands
    # From setup: n_kpts=174, nbnd=40
    # Check: 174 * 40 = 6960, but we have 7000 lines
    # Actually the .gnu file may have a different structure.
    # Let's figure it out: the x column should repeat for each band.
    # Find where x resets (goes back to a smaller value) to identify band boundaries.
    
    # Actually, looking at the data: x goes from 0 to ~5.7458, then resets.
    # Let's find the number of unique x values (k-points) and bands.
    
    # The x column is monotonically increasing within each band, then resets.
    # Find reset points: where x[i] < x[i-1]
    resets = np.where(x_col[1:] < x_col[:-1])[0] + 1
    n_bands = len(resets) + 1
    n_kpts = n_lines // n_bands
    
    print(f"Detected n_bands={n_bands}, n_kpts={n_kpts} from .gnu file")
    assert n_lines == n_bands * n_kpts, f"{n_lines} != {n_bands} * {n_kpts}"
    
    # Reshape into (n_bands, n_kpts) array
    energies = e_col.reshape(n_bands, n_kpts)
    x_unique = x_col[:n_kpts]  # x values for one band (same for all bands)
    
    # 5. Align to VBM
    # VBM is the maximum energy among occupied bands (bands 0 to n_occ-1)
    vbm = np.max(energies[:n_occ, :])
    print(f"VBM: {vbm:.4f} eV")
    
    # Shift all energies so VBM = 0
    energies_shifted = energies - vbm
    
    # 6. Find CBM (minimum of conduction bands, bands n_occ to end)
    cbm = np.min(energies_shifted[n_occ:, :])
    band_gap = cbm  # since VBM is at 0
    print(f"CBM: {cbm:.4f} eV")
    print(f"Band gap: {band_gap:.4f} eV")
    
    # 7. Build tick positions and labels
    # Use the high-symmetry points from bands_post.out (x in 2*pi/alat)
    # These are in the SAME units as x_col (2*pi/alat) -- INVARIANT 4
    tick_positions = [x for x, _ in hs_points]
    tick_labels = [label for _, label in hs_points]
    
    # INVARIANT 4 CHECK: tick positions must span approximately the same range
    # as the band data's x column
    assert abs(max(tick_positions) - x_col.max()) < 0.05 * x_col.max(), (
        f"Tick positions max={max(tick_positions):.4f} vs x_col max={x_col.max():.4f} -- unit mismatch!"
    )
    print(f"INVARIANT 4 check passed: tick max={max(tick_positions):.4f}, x_col max={x_col.max():.4f}")
    
    # 8. Detect discontinuities in the path
    # A discontinuity occurs where the x coordinate jumps (resets or jumps forward
    # without a continuous path). In the .gnu file, this shows up as a large
    # jump in x between consecutive k-points within the same band.
    # Actually, for a discontinuous path, the x coordinate jumps forward
    # (e.g., from Z at x=4.1914 to X at x=4.1914 -- same x but different k-point).
    # The .gnu file marks discontinuities by repeating the x value.
    # Let's detect: where x[i] == x[i-1] (zero distance increment) within a band.
    
    # Actually, looking at the high-symmetry points:
    # Z at x=4.1914 and X at x=4.1914 (same x, different k-point) -- this is a jump
    # R at x=4.9686 and M at x=4.9686 (same x, different k-point) -- this is a jump
    
    # For plotting, we need to split the line at these jumps.
    # Find indices where x repeats (consecutive same x values in x_unique)
    jump_indices = []
    for i in range(1, n_kpts):
        if abs(x_unique[i] - x_unique[i-1]) < 1e-10:
            jump_indices.append(i)
    
    print(f"Jump indices (discontinuities): {jump_indices}")
    
    # 9. Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each band, splitting at jumps
    for band_idx in range(n_bands):
        e = energies_shifted[band_idx, :]
        # Split into segments at jump points
        segments = []
        start = 0
        for j in jump_indices:
            segments.append((start, j + 1))  # include the jump point in both segments
            start = j
        segments.append((start, n_kpts))
        
        for seg_start, seg_end in segments:
            ax.plot(x_unique[seg_start:seg_end], e[seg_start:seg_end], 'b-', linewidth=0.8)
    
    # Add vertical lines at high-symmetry points
    for x, label in hs_points:
        ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Set x-axis ticks
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=10)
    
    # Set y-axis range: +/-10 eV around VBM (which is at 0)
    ax.set_ylim(-10, 10)
    ax.set_ylabel('Energy (eV)')
    ax.set_xlabel('k-path (2$\pi$/a)')
    ax.set_title(f'Rutile TiO$_2$ band structure (PBEsol)\nGap = {band_gap:.2f} eV')
    
    # Add a horizontal line at E=0 (VBM)
    ax.axhline(y=0, color='red', linestyle=':', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {OUTPUT_PDF}")
    
    # 10. Print summary
    print(f"\n=== SUMMARY ===")
    print(f"VBM: {vbm:.4f} eV (raw)")
    print(f"CBM: {vbm + cbm:.4f} eV (raw)")
    print(f"Band gap: {band_gap:.4f} eV")
    print(f"Number of electrons: {n_electrons}")
    print(f"Occupied bands: {n_occ}")
    print(f"Total bands plotted: {n_bands}")
    print(f"K-points: {n_kpts}")


if __name__ == '__main__':
    main()
