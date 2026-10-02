#!/usr/bin/env python3
"""
plot_bands.py: Plot the band structure for calculations/000.

Reads:
  - calculations/000/000.bands.dat (bands.x primary output: cartesian k-coords + energies)
  - calculations/000/band_labels.dat (high-symmetry labels and 0-based indices)
  - calculations/000/pw.out (to get 'number of electrons' for VBM alignment)

Writes:
  - calculations/000/bands_000.pdf

Prints:
  - VBM, CBM, and band gap (in eV)

Key details:
  - The .bands.dat file has cartesian k-coordinates (2*pi/alat), NOT cumulative distance.
    We compute cumulative path distance ourselves, adding NO distance across a
    discontinuity (detected via band_labels.dat labels containing '|').
  - VBM alignment: occupied bands = n_electrons / 2 (spin-unpolarized, non-metallic).
    VBM = max energy of the highest occupied band.
  - Y-axis: +/-10 eV around VBM (INVARIANT 4).
  - Ticks: at the x positions corresponding to each label's index, in the same
    cumulative-distance units as the band data (INVARIANT 4).
  - Discontinuities: no line drawn across the jump; tick labelled 'Z|X' etc.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import os

CALC_DIR = 'calculations/000'
BANDS_DAT = os.path.join(CALC_DIR, '000.bands.dat')
BAND_LABELS = os.path.join(CALC_DIR, 'band_labels.dat')
PW_OUT = os.path.join(CALC_DIR, 'pw.out')
OUTPUT_PDF = os.path.join(CALC_DIR, 'bands_000.pdf')


def parse_bands_dat(path):
    """
    Parse the bands.x primary output file (NNN.bands.dat).
    
    Format:
      Line 1: '&plot nbnd=NN, nks=NNN /'
      Then for each k-point:
        Line: kx ky kz  (cartesian coordinates in 2*pi/alat)
        Lines: nbnd energy values (in eV), spread over multiple lines
    
    Returns:
      kpts_cart: numpy array (nks, 3) -- cartesian k-coordinates
      energies: numpy array (nks, nbnd) -- band energies in eV
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    header = lines[0].strip()
    m = re.match(r'&plot\s+nbnd=\s*(\d+),\s*nks=\s*(\d+)\s*/', header)
    assert m, f"Cannot parse header: {header}"
    nbnd = int(m.group(1))
    nks = int(m.group(2))
    
    kpts_cart = np.zeros((nks, 3))
    energies = np.zeros((nks, nbnd))
    
    line_idx = 1
    for ik in range(nks):
        # Read k-point coordinates
        coords = lines[line_idx].split()
        assert len(coords) == 3, f"Expected 3 coords at line {line_idx+1}, got {len(coords)}"
        kpts_cart[ik] = [float(coords[0]), float(coords[1]), float(coords[2])]
        line_idx += 1
        
        # Read nbnd energy values (may span multiple lines)
        vals = []
        while len(vals) < nbnd:
            vals.extend(lines[line_idx].split())
            line_idx += 1
        assert len(vals) >= nbnd, f"Expected {nbnd} energies for k-point {ik}, got {len(vals)}"
        energies[ik] = [float(v) for v in vals[:nbnd]]
    
    assert line_idx <= len(lines), f"Read past end of file: line_idx={line_idx}, total={len(lines)}"
    
    return kpts_cart, energies


def parse_band_labels(path):
    """
    Parse band_labels.dat: each line is 'label index' (0-based index).
    Returns list of (label, index) sorted by index.
    """
    labels = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            assert len(parts) == 2, f"Expected 'label index', got: {line}"
            label = parts[0]
            idx = int(parts[1])
            labels.append((label, idx))
    labels.sort(key=lambda x: x[1])
    return labels


def get_n_electrons(pw_out_path):
    """
    Read 'number of electrons' from pw.out.
    """
    with open(pw_out_path, 'r') as f:
        for line in f:
            if 'number of electrons' in line:
                # Format: 'number of electrons       =        48.00'
                m = re.search(r'number of electrons\s*=\s*([\d.]+)', line)
                assert m, f"Cannot parse electron count from: {line}"
                return float(m.group(1))
    raise ValueError("'number of electrons' not found in pw.out")


def compute_cumulative_distance(kpts_cart, discontinuity_indices):
    """
    Compute cumulative path distance from cartesian k-coordinates.
    
    At a discontinuity (index in discontinuity_indices), NO distance is added
    from the previous point -- the x value stays the same.
    
    Args:
      kpts_cart: (nks, 3) array of cartesian k-coordinates
      discontinuity_indices: set of 0-based indices where a discontinuity occurs
                             (i.e., the k-point at this index is the START of a
                             new segment after a jump, so no distance is added
                             from the previous point to this one)
    
    Returns:
      x: (nks,) array of cumulative distances (non-decreasing)
    """
    nks = len(kpts_cart)
    x = np.zeros(nks)
    
    for i in range(1, nks):
        if i in discontinuity_indices:
            # Discontinuity: no distance added
            x[i] = x[i - 1]
        else:
            # Normal step: add Euclidean distance in cartesian k-space
            dk = kpts_cart[i] - kpts_cart[i - 1]
            x[i] = x[i - 1] + np.linalg.norm(dk)
    
    # INVARIANT 4: assert x never decreases
    diffs = np.diff(x)
    assert np.all(diffs >= -1e-12), \
        f"Cumulative distance decreases somewhere: min diff = {diffs.min()}"
    
    return x


def main():
    # --- Parse input files ---
    kpts_cart, energies = parse_bands_dat(BANDS_DAT)
    labels = parse_band_labels(BAND_LABELS)
    n_electrons = get_n_electrons(PW_OUT)
    
    nks, nbnd = energies.shape
    print(f"Parsed {BANDS_DAT}: nks={nks}, nbnd={nbnd}")
    print(f"Parsed {BAND_LABELS}: {len(labels)} labels")
    print(f"Number of electrons from pw.out: {n_electrons}")
    
    # --- VBM alignment (INVARIANT 2) ---
    # Spin-unpolarized, non-metallic: occupied bands = n_electrons / 2
    n_occ = int(n_electrons / 2)
    assert n_occ * 2 == int(n_electrons), \
        f"n_electrons={n_electrons} is not even -- cannot determine occupied bands"
    assert n_occ <= nbnd, \
        f"Occupied bands ({n_occ}) > nbnd ({nbnd}) -- nbnd is too small"
    
    # VBM = max energy of the highest occupied band (0-based index n_occ-1)
    vbm = np.max(energies[:, n_occ - 1])
    # CBM = min energy of the lowest conduction band (0-based index n_occ)
    cbm = np.min(energies[:, n_occ:])
    gap = cbm - vbm
    
    print(f"\nVBM = {vbm:.4f} eV")
    print(f"CBM = {cbm:.4f} eV")
    print(f"Band gap = {gap:.4f} eV")
    
    # Shift energies so VBM = 0
    energies_shifted = energies - vbm
    
    # --- Compute cumulative distance ---
    # Discontinuity indices: labels containing '|' mark the START of a new
    # segment after a jump. At these indices, no distance is added.
    discontinuity_indices = set()
    for label, idx in labels:
        if '|' in label:
            discontinuity_indices.add(idx)
    
    x = compute_cumulative_distance(kpts_cart, discontinuity_indices)
    
    # --- Verify tick positions span the same range as the band data (INVARIANT 4) ---
    tick_positions = [x[idx] for _, idx in labels]
    assert abs(max(tick_positions) - x.max()) < 0.05 * x.max(), \
        f"Tick positions ({min(tick_positions)} to {max(tick_positions)}) " \
        f"do not span the band data range (0 to {x.max()})"
    
    print(f"\nCumulative distance range: 0 to {x.max():.4f} (2*pi/alat)")
    print(f"Discontinuity indices: {sorted(discontinuity_indices)}")
    
    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bands, splitting at discontinuities so no line is drawn across jumps
    # Identify the segments: each discontinuity index starts a new segment
    # (the point at the discontinuity index is the first point of the new segment)
    
    # Build a list of (start_idx, end_idx) for each continuous segment
    # A segment ends just before a discontinuity index (exclusive)
    # and starts at 0 or at a discontinuity index (inclusive)
    
    # The discontinuity indices are where a NEW segment starts.
    # So the segments are:
    #   [0, d1), [d1, d2), [d2, ...)
    # where d1, d2, ... are the discontinuity indices.
    
    disc_sorted = sorted(discontinuity_indices)
    
    # Build segment boundaries
    boundaries = [0] + disc_sorted + [nks]
    
    for seg_i in range(len(boundaries) - 1):
        seg_start = boundaries[seg_i]
        seg_end = boundaries[seg_i + 1]  # exclusive
        
        # Plot all bands in this segment
        for ib in range(nbnd):
            ax.plot(x[seg_start:seg_end], energies_shifted[seg_start:seg_end, ib],
                    'b-', linewidth=0.5, alpha=0.7)
    
    # Add high-symmetry point ticks
    for label, idx in labels:
        xpos = x[idx]
        ax.axvline(x=xpos, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.text(xpos, -10.5, label, ha='center', va='top', fontsize=9)
    
    # Y-axis: +/-10 eV around VBM (INVARIANT 4)
    ax.set_ylim(-10, 10)
    ax.set_xlim(0, x.max())
    ax.set_xlabel('k-path (2$\pi$/a$_{lat}$)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('TiO$_2$ (rutile) band structure -- PBESOL, 6-atom primitive cell')
    ax.axhline(y=0, color='red', linestyle=':', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, dpi=150)
    print(f"\nSaved plot to {OUTPUT_PDF}")


if __name__ == '__main__':
    main()
