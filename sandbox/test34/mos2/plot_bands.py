import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# plot_bands.py
# Read the bands.x output (000.bands.dat) and band_labels.dat for calculations/000,
# align energies to the VBM, and plot the band structure.
# =============================================================================

CALC_DIR = 'calculations/000'
BANDS_FILE = os.path.join(CALC_DIR, '000.bands.dat')
LABELS_FILE = os.path.join(CALC_DIR, 'band_labels.dat')
PW_OUT = os.path.join(CALC_DIR, 'pw.out')
OUTPUT_PDF = os.path.join(CALC_DIR, 'bands_000.pdf')


def read_occupied_bands(pw_out_path):
    """Read the number of electrons from pw.out and return occupied band count."""
    with open(pw_out_path) as f:
        content = f.read()
    match = re.search(r'number of electrons\s*[:=]\s*([\d.]+)', content)
    if not match:
        match = re.search(r'number of electrons\s+([\d.]+)', content)
    if not match:
        raise ValueError("Could not find 'number of electrons' in pw.out")
    n_electrons = float(match.group(1))
    occupied_bands = int(n_electrons / 2)  # spin-unpolarized
    return occupied_bands


def parse_bands_dat(filepath):
    """
    Parse the primary filband file (NNN.bands.dat).
    Format: header '&plot nbnd=..., nks=... /', then for each k-point a block
    starting with 3 Cartesian coordinates (kx ky kz in 2*pi/alat), followed by
    nbnd energies in eV (10 per line, 4 lines for nbnd=40).
    Returns: kpts_cart (n_kpts, 3), energies (n_kpts, nbnd)
    """
    with open(filepath) as f:
        lines = f.readlines()
    
    # Find the header line
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('&plot'):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find '&plot' header in bands.dat")
    
    # Parse nbnd and nks from header
    header = lines[header_idx]
    nbnd_match = re.search(r'nbnd\s*=\s*(\d+)', header)
    nks_match = re.search(r'nks\s*=\s*(\d+)', header)
    if not nbnd_match or not nks_match:
        raise ValueError(f"Could not parse nbnd/nks from header: {header}")
    nbnd = int(nbnd_match.group(1))
    nks = int(nks_match.group(1))
    
    # Parse k-points and energies
    kpts_cart = np.zeros((nks, 3))
    energies = np.zeros((nks, nbnd))
    
    data_lines = lines[header_idx + 1:]
    # Remove blank lines
    data_lines = [line for line in data_lines if line.strip()]
    
    # Each k-point block: first line has 3 coords, then nbnd energy lines
    idx = 0
    for ik in range(nks):
        # Read k-point coordinates (3 floats)
        while idx < len(data_lines):
            parts = data_lines[idx].split()
            if len(parts) >= 3:
                try:
                    kx, ky, kz = float(parts[0]), float(parts[1]), float(parts[2])
                    kpts_cart[ik] = [kx, ky, kz]
                    idx += 1
                    break
                except ValueError:
                    idx += 1
        
        # Read energies (nbnd floats, possibly across multiple lines)
        n_read = 0
        while n_read < nbnd and idx < len(data_lines):
            parts = data_lines[idx].split()
            for p in parts:
                try:
                    energies[ik, n_read] = float(p)
                    n_read += 1
                    if n_read >= nbnd:
                        break
                except ValueError:
                    pass
            idx += 1
        
        if n_read < nbnd:
            raise ValueError(f"Could not read all {nbnd} energies for k-point {ik}")
    
    return kpts_cart, energies


def compute_cumulative_distance(kpts_cart, discontinuity_indices):
    """
    Compute cumulative path distance from consecutive k-points.
    No distance is added across a discontinuity.
    discontinuity_indices: list of k-point indices where the path jumps
    (i.e., the distance between kpts[i-1] and kpts[i] should NOT be added).
    Returns: x (n_kpts,) cumulative distance array
    """
    n_kpts = len(kpts_cart)
    x = np.zeros(n_kpts)
    
    # Mark discontinuity indices (where we should NOT add distance)
    disc_set = set(discontinuity_indices)
    
    for i in range(1, n_kpts):
        if i in disc_set:
            # Discontinuity: x[i] = x[i-1] (no distance added)
            x[i] = x[i - 1]
        else:
            # Add distance from previous k-point
            d = np.linalg.norm(kpts_cart[i] - kpts_cart[i - 1])
            x[i] = x[i - 1] + d
    
    # Assert x never decreases
    diffs = np.diff(x)
    assert np.all(diffs >= -1e-10), f"Cumulative distance decreases: min diff = {diffs.min()}"
    
    return x


def read_band_labels(filepath):
    """
    Read band_labels.dat: lines of "LABEL INDEX" or "LABEL1|LABEL2 INDEX".
    Returns: list of (label, index) pairs
    """
    labels = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                label = parts[0]
                index = int(parts[1])
                labels.append((label, index))
    return labels


def main():
    # Read occupied band count from pw.out
    occupied_bands = read_occupied_bands(PW_OUT)
    print(f"Occupied bands from pw.out: {occupied_bands}")
    
    # Parse bands data
    kpts_cart, energies = parse_bands_dat(BANDS_FILE)
    n_kpts, nbnd = energies.shape
    print(f"Parsed {n_kpts} k-points, {nbnd} bands")
    
    # Read band labels
    labels = read_band_labels(LABELS_FILE)
    print(f"Read {len(labels)} band labels")
    
    # Determine discontinuity indices from the k-point coordinates.
    # A discontinuity is where the distance between consecutive k-points is large
    # compared to the typical segment spacing (uniform interpolation within a segment).
    dists = np.zeros(n_kpts - 1)
    for i in range(n_kpts - 1):
        dists[i] = np.linalg.norm(kpts_cart[i + 1] - kpts_cart[i])
    
    nonzero_dists = dists[dists > 1e-10]
    if len(nonzero_dists) > 0:
        median_dist = np.median(nonzero_dists)
        threshold = 5 * median_dist
        discontinuity_indices = [i + 1 for i in range(n_kpts - 1) if dists[i] > threshold]
    else:
        discontinuity_indices = []
    
    print(f"Detected {len(discontinuity_indices)} discontinuities at indices: {discontinuity_indices}")
    
    # Compute cumulative distance
    x = compute_cumulative_distance(kpts_cart, discontinuity_indices)
    
    # Align energies to VBM
    # VBM is the maximum energy of the HIGHEST occupied band (band index occupied_bands-1)
    # across all k-points. This is NOT the max over all occupied bands (which would
    # include deep core-like bands).
    vbm = np.max(energies[:, occupied_bands - 1])
    energies_shifted = energies - vbm
    
    # CBM is the minimum energy of the LOWEST unoccupied band (band index occupied_bands)
    # across all k-points.
    cbm = np.min(energies[:, occupied_bands])
    band_gap = cbm - vbm
    
    print(f"VBM: {vbm:.4f} eV (raw)")
    print(f"CBM: {cbm:.4f} eV (raw)")
    print(f"Band gap: {band_gap:.4f} eV")
    
    # Build tick positions and labels
    # For each label, find its x position
    tick_positions = []
    tick_labels = []
    for label, idx in labels:
        if 0 <= idx < n_kpts:
            tick_positions.append(x[idx])
            tick_labels.append(label)
    
    # INVARIANT 4: assert tick positions span approximately the same range as band data
    if tick_positions:
        max_tick = max(tick_positions)
        max_x = x.max()
        assert abs(max_tick - max_x) < 0.05 * max_x, \
            f"Tick positions ({max_tick}) don't match band data range ({max_x})"
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bands, splitting at discontinuities
    # Create segments: each segment is a contiguous range of k-points without a discontinuity
    segments = []
    start = 0
    for disc_idx in discontinuity_indices:
        segments.append((start, disc_idx))  # [start, disc_idx) -- up to but not including disc_idx
        start = disc_idx
    segments.append((start, n_kpts))
    
    for band_idx in range(nbnd):
        for seg_start, seg_end in segments:
            x_seg = x[seg_start:seg_end]
            e_seg = energies_shifted[seg_start:seg_end, band_idx]
            ax.plot(x_seg, e_seg, 'b-', linewidth=0.8)
    
    # Add high-symmetry ticks
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    # Restrict y-axis to +/-10 eV around VBM
    ax.set_ylim(-10, 10)
    ax.set_xlabel('k-point path (2$\\pi$/a)')
    ax.set_ylabel('Energy (eV) relative to VBM')
    ax.set_title('2H MoS$_2$ Band Structure (PBEsol)')
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, dpi=150)
    print(f"Saved plot to {OUTPUT_PDF}")


if __name__ == '__main__':
    main()
