#!/usr/bin/env python3
"""
plot_bands.py: Parse the bands.x output (000.bands.dat) for calculations/000,
align energies to the VBM, and plot the band structure with proper high-symmetry
tick marks and discontinuity handling.

Format of 000.bands.dat (primary filband file):
  Line 1: &plot nbnd=40, nks=174 /
  Then for each k-point:
    Line: kx ky kz  (Cartesian coordinates in 2*pi/alat)
    4 lines of 10 energies each (40 bands total, in eV)

We compute the cumulative path distance ourselves from consecutive k-points,
adding NO distance across a discontinuity (where the previous segment's end
differs from the next segment's start).
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CALC_DIR = 'calculations/000'
BANDS_FILE = f'{CALC_DIR}/000.bands.dat'
LABELS_FILE = f'{CALC_DIR}/band_labels.dat'
PW_OUT = f'{CALC_DIR}/pw.out'
OUTPUT_PDF = f'{CALC_DIR}/bands_000.pdf'


def parse_bands_dat(path):
    """
    Parse the primary filband file (NNN.bands.dat).
    Returns:
        kpoints_cart: (n_kpts, 3) array of Cartesian k-coordinates (2*pi/alat)
        energies: (n_kpts, nbnd) array of band energies in eV
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    # Parse header: &plot nbnd=40, nks=174 /
    header = lines[0].strip()
    m = re.match(r'&plot\s+nbnd=\s*(\d+),\s*nks=\s*(\d+)\s*/', header)
    assert m, f"Cannot parse header: {header}"
    nbnd = int(m.group(1))
    nks = int(m.group(2))

    kpoints_cart = []
    energies = []

    i = 1
    for _ in range(nks):
        # k-point line: 3 floats
        k_line = lines[i].strip()
        k_vals = [float(x) for x in k_line.split()]
        assert len(k_vals) == 3, f"Expected 3 k-coordinates, got {len(k_vals)}: {k_line}"
        kpoints_cart.append(k_vals)
        i += 1

        # Energy lines: nbnd/10 lines of 10 energies each
        n_energy_lines = nbnd // 10
        band_energies = []
        for _ in range(n_energy_lines):
            e_line = lines[i].strip()
            e_vals = [float(x) for x in e_line.split()]
            band_energies.extend(e_vals)
            i += 1

        assert len(band_energies) == nbnd, \
            f"Expected {nbnd} energies, got {len(band_energies)}"
        energies.append(band_energies)

    kpoints_cart = np.array(kpoints_cart)
    energies = np.array(energies)

    assert kpoints_cart.shape == (nks, 3)
    assert energies.shape == (nks, nbnd)

    return kpoints_cart, energies


def compute_cumulative_distance(kpoints_cart, discontinuity_indices):
    """
    Compute cumulative path distance from consecutive k-points.
    At a discontinuity, add NO distance (the path jumps).
    
    Args:
        kpoints_cart: (n_kpts, 3) array of Cartesian k-coordinates
        discontinuity_indices: set of indices where a discontinuity occurs
            (i.e., the index of the k-point that starts a new segment after a jump)
    
    Returns:
        x: (n_kpts,) array of cumulative distances
    """
    n_kpts = len(kpoints_cart)
    x = np.zeros(n_kpts)

    for i in range(1, n_kpts):
        if i in discontinuity_indices:
            # Discontinuity: no distance added, x stays the same
            x[i] = x[i - 1]
        else:
            # Continuous: add the Euclidean distance between consecutive k-points
            dk = kpoints_cart[i] - kpoints_cart[i - 1]
            dist = np.linalg.norm(dk)
            x[i] = x[i - 1] + dist

    # Assert x never decreases
    diffs = np.diff(x)
    assert np.all(diffs >= -1e-10), \
        f"Cumulative distance decreases somewhere: min diff = {diffs.min()}"

    return x


def read_band_labels(path):
    """
    Read band_labels.dat: each line is 'LABEL INDEX'.
    Returns list of (label, index) tuples.
    """
    labels = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            assert len(parts) == 2, f"Expected 'LABEL INDEX', got: {line}"
            label = parts[0]
            idx = int(parts[1])
            labels.append((label, idx))
    return labels


def read_occupied_bands(pw_out_path):
    """
    Read the number of Kohn-Sham states (occupied bands) from pw.out.
    """
    with open(pw_out_path, 'r') as f:
        for line in f:
            if 'number of Kohn-Sham states' in line:
                # Format: "number of Kohn-Sham states=           24"
                m = re.search(r'number of Kohn-Sham states\s*=\s*(\d+)', line)
                assert m, f"Cannot parse Kohn-Sham states from: {line}"
                return int(m.group(1))
    raise ValueError("Could not find 'number of Kohn-Sham states' in pw.out")


def main():
    # Parse the bands data
    kpoints_cart, energies = parse_bands_dat(BANDS_FILE)
    n_kpts, nbnd = energies.shape
    print(f"Parsed {BANDS_FILE}: {n_kpts} k-points, {nbnd} bands")

    # Read band labels
    labels = read_band_labels(LABELS_FILE)
    print(f"Read {len(labels)} band labels from {LABELS_FILE}")

    # Identify discontinuities from the labels
    # A discontinuity occurs at index i if there's a label with '|' at that index
    # and the previous segment's end differs from this segment's start
    discontinuity_indices = set()
    for label, idx in labels:
        if '|' in label:
            discontinuity_indices.add(idx)

    print(f"Discontinuity indices: {sorted(discontinuity_indices)}")

    # Compute cumulative path distance
    x = compute_cumulative_distance(kpoints_cart, discontinuity_indices)
    print(f"Cumulative distance range: {x[0]:.4f} to {x[-1]:.4f} (2*pi/alat)")

    # Read occupied band count from pw.out
    n_occ = read_occupied_bands(PW_OUT)
    print(f"Occupied bands (from pw.out): {n_occ}")
    assert n_occ <= nbnd, f"Occupied bands ({n_occ}) > nbnd ({nbnd})"

    # Find VBM: maximum energy among occupied bands
    # Occupied bands are indices 0 to n_occ-1
    vbm = np.max(energies[:, :n_occ])
    print(f"VBM: {vbm:.4f} eV")

    # Find CBM: minimum energy among unoccupied bands
    cbm = np.min(energies[:, n_occ:])
    print(f"CBM: {cbm:.4f} eV")

    # Band gap
    band_gap = cbm - vbm
    print(f"Band gap: {band_gap:.4f} eV")

    # Align energies to VBM
    energies_aligned = energies - vbm

    # Build tick positions and labels
    # For each label, the tick position is x[idx]
    tick_positions = []
    tick_labels = []
    for label, idx in labels:
        tick_positions.append(x[idx])
        tick_labels.append(label)

    # INVARIANT 4 CHECK: tick positions must span approximately the same range
    # as the band data's x column
    assert abs(max(tick_positions) - x.max()) < 0.05 * x.max(), \
        f"Tick positions ({min(tick_positions)} to {max(tick_positions)}) " \
        f"do not match band data x range ({x.min()} to {x.max()})"

    print(f"Tick positions: {tick_positions}")
    print(f"Tick labels: {tick_labels}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each band, splitting at discontinuities
    for band_idx in range(nbnd):
        band_energies = energies_aligned[:, band_idx]

        # Find where to split the line (at discontinuities)
        # We need to break the line at each discontinuity index
        segments = []
        start = 0
        for disc_idx in sorted(discontinuity_indices):
            if disc_idx > start:
                segments.append((start, disc_idx))
            start = disc_idx
        if start < n_kpts:
            segments.append((start, n_kpts))

        for seg_start, seg_end in segments:
            ax.plot(x[seg_start:seg_end], band_energies[seg_start:seg_end],
                    'b-', linewidth=0.8)

    # Set x-axis ticks
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, ha='center')

    # Set y-axis range: +/-10 eV around VBM (which is now at 0)
    ax.set_ylim(-10, 10)

    # Labels and title
    ax.set_xlabel('k-point path (2$\\pi$/a)')
    ax.set_ylabel('Energy (eV) relative to VBM')
    ax.set_title('Rutile TiO$_2$ Band Structure (PBEsol, SCF)')

    # Add a horizontal line at E=0 (VBM)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_PDF, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {OUTPUT_PDF}")

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"VBM: {vbm:.4f} eV")
    print(f"CBM: {cbm:.4f} eV")
    print(f"Band gap: {band_gap:.4f} eV")
    print(f"Occupied bands: {n_occ}")
    print(f"Total bands: {nbnd}")
    print(f"K-points: {n_kpts}")
    print(f"Discontinuities at indices: {sorted(discontinuity_indices)}")


if __name__ == '__main__':
    main()
