#!/usr/bin/env python
"""Plot the band structure for wurtzite ZnO (calculations/000).

Reads:
  - calculations/000/000.bands.dat (primary filband from bands.x)
  - calculations/000/band_labels.dat (high-symmetry labels and indices)
  - calculations/000/discontinuities.dat (indices where path is discontinuous)
  - calculations/000/pw.out (to get the number of occupied bands for VBM alignment)

Writes:
  - calculations/000/bands_000.pdf

Prints:
  - VBM, CBM, and band gap (in eV)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re

CALC_DIR = 'calculations/000'
BANDS_FILE = f'{CALC_DIR}/000.bands.dat'
LABELS_FILE = f'{CALC_DIR}/band_labels.dat'
DISC_FILE = f'{CALC_DIR}/discontinuities.dat'
PW_OUT = f'{CALC_DIR}/pw.out'
OUTPUT_PDF = f'{CALC_DIR}/bands_000.pdf'


def parse_bands_dat(filename):
    """Parse the primary filband file from bands.x.
    
    Format:
      &plot nbnd=40, nks=360 /
      kx ky kz  (Cartesian coords in 2*pi/alat)
      e1 e2 ... e10
      e11 e12 ... e20
      e21 e22 ... e30
      e31 e32 ... e40
      (next k-point block)
      ...
    
    Returns:
        kpoints: (nks, 3) array of Cartesian k-coordinates in 2*pi/alat
        energies: (nks, nbnd) array of band energies in eV
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    header = lines[0].strip()
    match = re.match(r'&plot\s+nbnd=\s*(\d+),\s*nks=\s*(\d+)\s*/', header)
    assert match, f"Could not parse header: {header}"
    nbnd = int(match.group(1))
    nks = int(match.group(2))
    
    kpoints = np.zeros((nks, 3))
    energies = np.zeros((nks, nbnd))
    
    line_idx = 1
    for ik in range(nks):
        # Read k-point coordinates (3 values on one line)
        k_line = lines[line_idx].strip()
        kvals = k_line.split()
        assert len(kvals) == 3, f"Expected 3 k-coords, got {len(kvals)}: {k_line}"
        kpoints[ik] = [float(kvals[0]), float(kvals[1]), float(kvals[2])]
        line_idx += 1
        
        # Read energies (nbnd values, 10 per line)
        n_lines = nbnd // 10
        if nbnd % 10 != 0:
            n_lines += 1
        
        for i in range(n_lines):
            e_line = lines[line_idx].strip()
            evals = e_line.split()
            start = i * 10
            end = min(start + 10, nbnd)
            energies[ik, start:end] = [float(e) for e in evals[:end-start]]
            line_idx += 1
    
    assert line_idx <= len(lines), f"Read past end of file: line_idx={line_idx}, total={len(lines)}"
    return kpoints, energies


def compute_cumulative_distance(kpoints, discontinuities):
    """Compute cumulative path distance from Cartesian k-coordinates.
    
    No distance is added across a discontinuity (the path jumps).
    
    Args:
        kpoints: (nks, 3) array of Cartesian k-coordinates in 2*pi/alat
        discontinuities: list of indices where the path is discontinuous
                         (i.e., kpoints[i] is the START of a new segment
                          that doesn't connect to kpoints[i-1])
    
    Returns:
        x: (nks,) array of cumulative path distances in 2*pi/alat
    """
    nks = len(kpoints)
    x = np.zeros(nks)
    
    # Build a set of discontinuity indices for fast lookup
    disc_set = set(discontinuities)
    
    for i in range(1, nks):
        if i in disc_set:
            # Discontinuity: no distance added, x stays the same
            x[i] = x[i-1]
        else:
            # Continuous: add the distance between consecutive k-points
            dk = np.linalg.norm(kpoints[i] - kpoints[i-1])
            x[i] = x[i-1] + dk
    
    # INVARIANT 4: assert x never decreases
    assert np.all(np.diff(x) >= -1e-10), "Cumulative distance decreased somewhere!"
    
    return x


def read_labels(filename):
    """Read band_labels.dat: lines of 'LABEL INDEX'.
    
    Returns:
        labels: list of (label, index) tuples
    """
    labels = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            assert len(parts) == 2, f"Expected 'LABEL INDEX', got: {line}"
            labels.append((parts[0], int(parts[1])))
    return labels


def read_discontinuities(filename):
    """Read discontinuities.dat: one index per line."""
    discs = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            discs.append(int(line))
    return discs


def get_occupied_bands(pw_out_file):
    """Parse pw.out to get the number of occupied bands (Kohn-Sham states)."""
    with open(pw_out_file, 'r') as f:
        for line in f:
            if 'number of Kohn-Sham states' in line:
                # Format: "     number of Kohn-Sham states=           26"
                match = re.search(r'number of Kohn-Sham states=\s*(\d+)', line)
                assert match, f"Could not parse Kohn-Sham states from: {line}"
                return int(match.group(1))
    raise ValueError("Could not find 'number of Kohn-Sham states' in pw.out")


def main():
    # Parse the bands data
    kpoints, energies = parse_bands_dat(BANDS_FILE)
    nks, nbnd = energies.shape
    print(f"Parsed {nks} k-points, {nbnd} bands")
    
    # Read labels and discontinuities
    labels = read_labels(LABELS_FILE)
    discontinuities = read_discontinuities(DISC_FILE)
    print(f"Labels: {labels}")
    print(f"Discontinuities at indices: {discontinuities}")
    
    # Compute cumulative path distance
    x = compute_cumulative_distance(kpoints, discontinuities)
    print(f"Cumulative distance range: {x[0]:.4f} to {x[-1]:.4f} (2*pi/alat)")
    
    # Get occupied bands from pw.out
    nocc = get_occupied_bands(PW_OUT)
    print(f"Number of occupied bands (from pw.out): {nocc}")
    
    # INVARIANT 2: VBM is the maximum of the occupied bands (index nocc-1)
    # CBM is the minimum of the unoccupied bands (index nocc onwards)
    vbm = np.max(energies[:, :nocc])
    cbm = np.min(energies[:, nocc:])
    gap = cbm - vbm
    print(f"VBM = {vbm:.4f} eV")
    print(f"CBM = {cbm:.4f} eV")
    print(f"Band gap = {gap:.4f} eV")
    
    # Align energies to VBM (subtract VBM so VBM is at 0)
    energies_aligned = energies - vbm
    
    # Build tick positions and labels
    # For each label, the tick position is x[index]
    tick_positions = []
    tick_labels = []
    for label, idx in labels:
        tick_positions.append(x[idx])
        tick_labels.append(label)
    
    # INVARIANT 4: assert tick positions span approximately the same range as x
    assert abs(max(tick_positions) - x[-1]) < 0.05 * x[-1], \
        f"Tick positions ({max(tick_positions):.4f}) don't match x range ({x[-1]:.4f})"
    
    # Handle discontinuities: label with both sides ('Z|X') and draw no line across
    # We need to identify which labels are at discontinuity points
    # A discontinuity at index i means kpoints[i] is the start of a new segment
    # The label at index i-1 is the end of the previous segment
    # The label at index i is the start of the new segment
    
    # Build a dict: index -> label
    label_at_index = {}
    for label, idx in labels:
        label_at_index[idx] = label
    
    # For each discontinuity, check if both sides have labels
    # and modify the tick label to show both sides
    final_tick_positions = []
    final_tick_labels = []
    
    # We need to handle the case where a discontinuity index has a label
    # and the previous index also has a label
    # In that case, we combine them as 'prev|new'
    
    # First, let's identify which tick positions correspond to discontinuities
    # A discontinuity at index i means the path jumps from kpoints[i-1] to kpoints[i]
    # The tick at x[i] should be labeled with both the end of the previous segment
    # and the start of the new segment
    
    # Let's rebuild the ticks, handling discontinuities
    for i, (label, idx) in enumerate(labels):
        if idx in discontinuities:
            # This is a discontinuity point
            # Check if the previous index (idx-1) also has a label
            if (idx - 1) in label_at_index:
                prev_label = label_at_index[idx - 1]
                combined_label = f"{prev_label}|{label}"
                final_tick_positions.append(x[idx])
                final_tick_labels.append(combined_label)
            else:
                final_tick_positions.append(x[idx])
                final_tick_labels.append(label)
        else:
            final_tick_positions.append(x[idx])
            final_tick_labels.append(label)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each band, splitting at discontinuities
    for ib in range(nbnd):
        # Create a mask for where to draw lines
        # We draw lines between consecutive k-points, but NOT across discontinuities
        for i in range(nks - 1):
            if (i + 1) in set(discontinuities):
                # Don't draw a line across this discontinuity
                continue
            ax.plot([x[i], x[i+1]], [energies_aligned[i, ib], energies_aligned[i+1, ib]],
                    'b-', linewidth=0.5)
    
    # Set x-axis ticks
    ax.set_xticks(final_tick_positions)
    ax.set_xticklabels(final_tick_labels)
    
    # Set y-axis range: +/-10 eV around VBM (which is now at 0)
    ax.set_ylim(-10, 10)
    
    # Labels and title
    ax.set_xlabel('k-path (2$\pi$/a$_{lat}$)')
    ax.set_ylabel('Energy (eV) relative to VBM')
    ax.set_title('Wurtzite ZnO Band Structure (PBEsol)')
    
    # Add a horizontal line at E=0 (VBM)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, dpi=150)
    print(f"Saved plot to {OUTPUT_PDF}")


if __name__ == '__main__':
    main()
