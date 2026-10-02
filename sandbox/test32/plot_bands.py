#!/usr/bin/env python3
"""
plot_bands.py: Plot the band structure for calculations/000.

Reads:
  - calculations/000/000.bands.dat (primary filband file from bands.x)
    Format: header '&plot nbnd=..., nks=... /' then for each k-point:
      a line with 3 cartesian k-coordinates (kx ky kz in 2*pi/alat)
      followed by nbnd lines of energies (one per band, in eV)
  - calculations/000/band_labels.dat (label and 0-based index for each
    high-symmetry point)
  - calculations/000/pw.out (to read 'number of electrons' for VBM alignment)

Steps:
  1. Parse 000.bands.dat: extract k-points (cartesian) and band energies.
  2. Compute cumulative path distance from consecutive k-points.
     At a discontinuity (jump in k-space), add NO distance (x stays the same
     for the new segment's start point, so the plot shows a vertical gap).
     Assert x values never decrease.
  3. Read 'number of electrons' from pw.out -> occupied bands = n_elec/2.
     VBM = highest occupied band energy (band index = occupied-1, 0-based).
  4. Align all energies to VBM (subtract VBM energy).
  5. Read band_labels.dat for tick positions (0-based index -> x coordinate).
  6. Plot with matplotlib: y range +/-10 eV around VBM, ticks at high-symmetry
     points, no line across discontinuities.
  7. Save to calculations/000/bands_000.pdf.
  8. Print VBM, CBM, and band gap.
"""

import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CALC_DIR = 'calculations/000'
BANDS_DAT = f'{CALC_DIR}/000.bands.dat'
BAND_LABELS = f'{CALC_DIR}/band_labels.dat'
PW_OUT = f'{CALC_DIR}/pw.out'
OUTPUT_PDF = f'{CALC_DIR}/bands_000.pdf'


def parse_bands_dat(fname):
    """
    Parse the primary filband file (NNN.bands.dat).
    
    Format:
      &plot nbnd=  48, nks=   183 /
      kx ky kz
      E1 E2 ... E48   (one line per k-point, 48 energies)
      kx ky kz
      E1 E2 ... E48
      ...
    
    Returns:
      kpts_cart: numpy array (n_kpts, 3) -- cartesian k-coordinates (2*pi/alat)
      energies: numpy array (n_kpts, nbnd) -- band energies in eV
    """
    with open(fname, 'r') as f:
        lines = f.readlines()
    
    # Find the header line
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('&plot'):
            header_idx = i
            break
    assert header_idx is not None, f"Could not find '&plot' header in {fname}"
    
    # Parse nbnd and nks from header (allow spaces after '=')
    header = lines[header_idx]
    nbnd_match = re.search(r'nbnd=\s*(\d+)', header)
    nks_match = re.search(r'nks=\s*(\d+)', header)
    assert nbnd_match and nks_match, f"Could not parse nbnd/nks from header: {header}"
    nbnd = int(nbnd_match.group(1))
    nks = int(nks_match.group(1))
    
    # Parse k-points and energies
    kpts_cart = np.zeros((nks, 3))
    energies = np.zeros((nks, nbnd))
    
    data_lines = lines[header_idx + 1:]
    # Remove blank lines
    data_lines = [l for l in data_lines if l.strip()]
    
    # Each k-point block: 1 line of k-coords + 1 line of nbnd energies
    # (or possibly multiple lines of energies if nbnd is large)
    # Let's parse more carefully: after the header, the format is:
    #   kx ky kz
    #   E1 E2 ... E_nbnd
    #   kx ky kz
    #   E1 E2 ... E_nbnd
    #   ...
    # But energies might be split across multiple lines if nbnd is large.
    # Let's read all remaining numbers and group them.
    
    all_numbers = []
    for line in data_lines:
        nums = line.split()
        for n in nums:
            try:
                all_numbers.append(float(n))
            except ValueError:
                pass
    
    # Total numbers should be nks * (3 + nbnd)
    expected = nks * (3 + nbnd)
    assert len(all_numbers) >= expected, \
        f"Expected {expected} numbers, got {len(all_numbers)}"
    
    for i in range(nks):
        base = i * (3 + nbnd)
        kpts_cart[i] = all_numbers[base:base+3]
        energies[i] = all_numbers[base+3:base+3+nbnd]
    
    return kpts_cart, energies


def compute_cumulative_distance(kpts_cart):
    """
    Compute cumulative path distance from cartesian k-coordinates.
    
    At a discontinuity (where the k-point jumps to a new segment start),
    we add NO distance -- the x value stays the same, creating a vertical
    gap in the plot.
    
    We detect discontinuities by checking if the distance between consecutive
    k-points is anomalously large compared to the typical step size.
    
    Returns:
      x: numpy array (n_kpts,) -- cumulative path distance (2*pi/alat)
    """
    n_kpts = len(kpts_cart)
    x = np.zeros(n_kpts)
    
    # Compute all consecutive distances
    dists = np.zeros(n_kpts - 1)
    for i in range(n_kpts - 1):
        dists[i] = np.linalg.norm(kpts_cart[i+1] - kpts_cart[i])
    
    # Typical step size: median of the smallest distances
    # (interior points are evenly spaced, so most distances are similar)
    # Use the 10th percentile as a robust estimate of the step size
    step_size = np.percentile(dists, 10)
    
    # A discontinuity is where the distance is much larger than the step size
    # (e.g., > 3x the step size)
    discontinuity_threshold = 3.0 * step_size
    
    for i in range(n_kpts - 1):
        if dists[i] > discontinuity_threshold:
            # Discontinuity: do NOT add distance
            x[i+1] = x[i]
        else:
            x[i+1] = x[i] + dists[i]
    
    # Assert x never decreases
    assert np.all(np.diff(x) >= -1e-10), \
        f"Cumulative distance decreases somewhere: min diff = {np.min(np.diff(x))}"
    
    return x


def read_number_of_electrons(pw_out_path):
    """
    Read 'number of electrons' from pw.out.
    Returns the number of electrons (float).
    """
    with open(pw_out_path, 'r') as f:
        for line in f:
            if 'number of electrons' in line:
                # Format: 'number of electrons       =        48.00'
                match = re.search(r'number of electrons\s*=\s*([\d.]+)', line)
                assert match, f"Could not parse number of electrons from: {line}"
                return float(match.group(1))
    raise ValueError(f"Could not find 'number of electrons' in {pw_out_path}")


def read_band_labels(fname):
    """
    Read band_labels.dat: each line is 'label index' (0-based index).
    Returns list of (label, index) tuples.
    """
    labels = []
    with open(fname, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                label = parts[0]
                idx = int(parts[1])
                labels.append((label, idx))
    return labels


def main():
    # 1. Parse bands data
    kpts_cart, energies = parse_bands_dat(BANDS_DAT)
    n_kpts, nbnd = energies.shape
    print(f"Parsed {n_kpts} k-points, {nbnd} bands from {BANDS_DAT}")
    
    # 2. Compute cumulative distance
    x = compute_cumulative_distance(kpts_cart)
    print(f"Cumulative distance range: {x[0]:.4f} to {x[-1]:.4f} (2*pi/alat)")
    
    # 3. Read number of electrons from pw.out
    n_elec = read_number_of_electrons(PW_OUT)
    occupied_bands = int(n_elec / 2)  # spin-unpolarized
    print(f"Number of electrons: {n_elec}, occupied bands: {occupied_bands}")
    
    # VBM: highest occupied band (band index = occupied_bands - 1, 0-based)
    # Find the maximum energy among occupied bands at all k-points
    vbm = np.max(energies[:, :occupied_bands])
    print(f"VBM (raw): {vbm:.4f} eV")
    
    # CBM: lowest unoccupied band (band index = occupied_bands, 0-based)
    cbm = np.min(energies[:, occupied_bands:])
    print(f"CBM (raw): {cbm:.4f} eV")
    
    # Band gap
    band_gap = cbm - vbm
    print(f"Band gap: {band_gap:.4f} eV")
    
    # 4. Align energies to VBM
    energies_aligned = energies - vbm
    
    # 5. Read band labels
    labels = read_band_labels(BAND_LABELS)
    print(f"Band labels: {labels}")
    
    # Map labels to x positions
    tick_positions = []
    tick_labels = []
    for label, idx in labels:
        assert 0 <= idx < n_kpts, f"Label index {idx} out of range [0, {n_kpts})"
        tick_positions.append(x[idx])
        tick_labels.append(label)
    
    # INVARIANT 4: assert tick positions span approximately the same range
    # as the band data's x column
    assert abs(max(tick_positions) - x[-1]) < 0.05 * x[-1], \
        f"Tick positions ({max(tick_positions):.4f}) don't match x range ({x[-1]:.4f})"
    
    # 6. Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot all bands, but restrict y-axis to +/-10 eV around VBM
    y_min = -10.0
    y_max = 10.0
    
    # We need to split the plot at discontinuities (where x doesn't increase)
    # Find discontinuity indices
    disc_indices = []
    for i in range(1, n_kpts):
        if x[i] - x[i-1] < 1e-10:  # no distance added -> discontinuity
            disc_indices.append(i)
    
    # Build segments: list of (start_idx, end_idx) inclusive
    segments = []
    seg_start = 0
    for i in disc_indices:
        segments.append((seg_start, i - 1))
        seg_start = i
    segments.append((seg_start, n_kpts - 1))
    
    for band_idx in range(nbnd):
        for seg_start, seg_end in segments:
            x_seg = x[seg_start:seg_end+1]
            e_seg = energies_aligned[seg_start:seg_end+1, band_idx]
            # Only plot if any energy is in the y range
            if np.any((e_seg >= y_min) & (e_seg <= y_max)):
                ax.plot(x_seg, e_seg, 'b-', linewidth=0.5)
    
    # Add tick marks at high-symmetry points
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    # Draw vertical lines at discontinuities
    for i in disc_indices:
        ax.axvline(x=x[i], color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Labels and formatting
    ax.set_xlabel(r'k-path (2$\pi$/a)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title(r'TiO$_2$ (rutile) Band Structure - PBEsol')
    ax.set_ylim(y_min, y_max)
    ax.axhline(y=0, color='red', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {OUTPUT_PDF}")
    
    # 7. Print summary
    print(f"\n=== Band Structure Summary ===")
    print(f"VBM: {vbm:.4f} eV (raw)")
    print(f"CBM: {cbm:.4f} eV (raw)")
    print(f"Band gap: {band_gap:.4f} eV")
    print(f"Occupied bands: {occupied_bands}")
    print(f"Total bands: {nbnd}")
    print(f"K-points: {n_kpts}")
    print(f"Discontinuities at indices: {disc_indices}")


if __name__ == '__main__':
    main()
