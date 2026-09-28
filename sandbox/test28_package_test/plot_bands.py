#!/usr/bin/env python3
"""
Plot the band structure for calculations/000 (rutile TiO2, unperturbed).

Reads:
  - calculations/000/000.bands.dat.gnu  (bands.x output, one block per band)
  - calculations/000/band_labels.dat    (high-symmetry labels and k-point indices)
  - calculations/000/pw.out             (to get the number of electrons / occupied bands)

Writes:
  - calculations/000/bands_000.pdf

Prints VBM, CBM, and band gap (eV, relative to VBM).
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CALC_DIR = 'calculations/000'
GNU_FILE = f'{CALC_DIR}/000.bands.dat.gnu'
LABELS_FILE = f'{CALC_DIR}/band_labels.dat'
PW_OUT = f'{CALC_DIR}/pw.out'
OUTPUT_PDF = f'{CALC_DIR}/bands_000.pdf'


def read_occupied_bands(pw_out_path):
    """Read the number of Kohn-Sham states (occupied bands) from pw.out."""
    with open(pw_out_path) as f:
        content = f.read()
    # Look for 'number of Kohn-Sham states = N'
    m = re.search(r'number of Kohn-Sham states\s*=\s*(\d+)', content)
    assert m, f"Could not find 'number of Kohn-Sham states' in {pw_out_path}"
    n_occ = int(m.group(1))
    print(f"Occupied bands (from pw.out): {n_occ}")
    return n_occ


def read_gnu_bands(gnu_path, nbnd):
    """
    Read the .gnu file: nbnd blocks, each with (x, energy) pairs.
    Returns:
      x: (nkpts,) array of cumulative path distance in 2*pi/alat
      energies: (nbnd, nkpts) array of energies in eV
    """
    with open(gnu_path) as f:
        lines = f.readlines()

    # Parse into blocks separated by blank lines
    blocks = []
    current = []
    for line in lines:
        line = line.strip()
        if line == '':
            if current:
                blocks.append(current)
                current = []
        else:
            parts = line.split()
            if len(parts) >= 2:
                current.append((float(parts[0]), float(parts[1])))
    if current:
        blocks.append(current)

    assert len(blocks) == nbnd, f"Expected {nbnd} blocks, got {len(blocks)}"

    # All blocks should have the same number of points
    nkpts = len(blocks[0])
    for i, b in enumerate(blocks):
        assert len(b) == nkpts, f"Block {i} has {len(b)} points, expected {nkpts}"

    x = np.array([p[0] for p in blocks[0]])
    energies = np.array([[p[1] for p in b] for b in blocks])

    return x, energies


def read_labels(labels_path):
    """
    Read band_labels.dat: lines of 'label\tindex'.
    Returns list of (label, index) tuples.
    """
    labels = []
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            assert len(parts) == 2, f"Bad line in {labels_path}: {line}"
            labels.append((parts[0], int(parts[1])))
    return labels


def find_vbm_cbm(energies, n_occ):
    """
    Find VBM and CBM using the physical definition:
    - VBM = highest energy among occupied bands (bands 0 to n_occ-1)
    - CBM = lowest energy among unoccupied bands (bands n_occ to nbnd-1)
    
    The bands in the .gnu file are ordered by energy at each k-point.
    With semicore pseudopotentials, the occupied bands include semicore states
    (Ti 3s/3p, O 2s) which are at very low energies. The valence bands are
    the highest occupied bands, and the conduction bands start right after.
    
    For an insulator, there is a clear gap between the highest occupied and
    lowest unoccupied bands. We use n_occ from pw.out to split the bands.
    """
    nbnd = energies.shape[0]
    
    # VBM = max energy among occupied bands
    vbm = np.max(energies[:n_occ, :])
    # CBM = min energy among unoccupied bands
    cbm = np.min(energies[n_occ:, :])
    gap = cbm - vbm
    
    return vbm, cbm, gap


def main():
    # Read occupied band count from pw.out (INVARIANT 2)
    n_occ = read_occupied_bands(PW_OUT)

    # Read band data from .gnu file
    nbnd = 40  # must match nbnd in bands.in
    x, energies = read_gnu_bands(GNU_FILE, nbnd)
    nkpts = len(x)
    print(f"Read {nbnd} bands x {nkpts} k-points from {GNU_FILE}")

    # Read high-symmetry labels
    labels = read_labels(LABELS_FILE)
    print(f"Read {len(labels)} high-symmetry labels from {LABELS_FILE}")

    # INVARIANT 4: tick positions must be in the same units as the band data x column.
    # The labels give indices into the k-point list; the .gnu x column is cumulative
    # path distance in 2*pi/alat. So tick positions = x[label_index].
    tick_positions = [x[idx] for _, idx in labels]
    tick_labels = [lbl for lbl, _ in labels]

    # Assert tick positions span approximately the same range as the band data x column
    assert abs(max(tick_positions) - x.max()) < 0.05 * x.max(), \
        f"Tick positions max={max(tick_positions)} vs x.max()={x.max()} -- unit mismatch!"
    assert abs(min(tick_positions) - x.min()) < 0.05 * x.max(), \
        f"Tick positions min={min(tick_positions)} vs x.min()={x.min()} -- unit mismatch!"

    # Find VBM and CBM using the physical definition
    vbm, cbm, gap = find_vbm_cbm(energies, n_occ)

    print(f"VBM = {vbm:.4f} eV")
    print(f"CBM = {cbm:.4f} eV")
    print(f"Band gap = {gap:.4f} eV")

    # Shift energies so VBM is at 0
    energies_shifted = energies - vbm

    # Plot: restrict y-axis to +/-10 eV around VBM (INVARIANT 4)
    emin, emax = -10.0, 10.0

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each band, splitting at discontinuities (where x doesn't increase)
    # A discontinuity is marked by a repeated x value (zero distance increment)
    for band_idx in range(nbnd):
        e = energies_shifted[band_idx, :]
        # Only plot bands within the y-range (with a small margin)
        if np.max(e) < emin or np.min(e) > emax:
            continue

        # Find discontinuities: where x[i] == x[i-1] (repeated x value)
        # Split the line at these points
        segments = []
        seg_start = 0
        for i in range(1, nkpts):
            if x[i] == x[i - 1]:  # discontinuity: x doesn't increase
                segments.append((seg_start, i))
                seg_start = i  # start new segment at the repeated point
        segments.append((seg_start, nkpts))

        for (s, e_idx) in segments:
            ax.plot(x[s:e_idx], e[s:e_idx], 'b-', linewidth=0.8)

    # Add high-symmetry ticks
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, ha='center')

    # Add vertical lines at high-symmetry points
    for pos in tick_positions:
        ax.axvline(x=pos, color='gray', linewidth=0.3, alpha=0.5)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(emin, emax)
    ax.set_xlabel('k-point path (2$\\pi$/a)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Rutile TiO$_2$ band structure (PBEsol, nbnd=40)')
    ax.axhline(y=0, color='red', linewidth=0.5, linestyle='--', label='VBM')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, dpi=150)
    print(f"Saved plot to {OUTPUT_PDF}")


if __name__ == '__main__':
    main()
