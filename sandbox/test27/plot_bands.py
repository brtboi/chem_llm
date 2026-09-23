#!/usr/bin/env python
"""
Plot the band structure for calculations/000 (rutile TiO2, unperturbed base).

Reads:
  - calculations/000/000.bands.dat.gnu  (bands.x output: one block per band, each block is (k_distance, energy) for n_kpts lines, blocks separated by blank lines)
  - calculations/000/band_labels.dat    (high-symmetry labels and their k-point indices)
  - calculations/000/pw.out             (to get 'number of electrons' and 'number of Kohn-Sham states' for VBM alignment)
  - calculations/000/bands.in           (to get nbnd for sanity check)
  - calculations/000/bands_pw.out       (to get 'number of k points' for sanity check)

Writes:
  - calculations/000/bands_000.pdf

Prints the VBM, CBM and band gap in eV.

INVARIANT 2: The occupied-band count is read from pw.out ('number of Kohn-Sham states'),
not assumed from a per-element valence count. VBM = max energy of the last occupied band.

INVARIANT 4: The x-axis uses the cumulative k-path distance from the .gnu file for both
the band curves and the high-symmetry tick positions. We verify that the tick positions
span approximately the same range as the band data's k column before plotting.
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CALC_DIR = 'calculations/000'
BANDS_GNU = f'{CALC_DIR}/000.bands.dat.gnu'
BAND_LABELS = f'{CALC_DIR}/band_labels.dat'
PW_OUT = f'{CALC_DIR}/pw.out'
BANDS_IN = f'{CALC_DIR}/bands.in'
BANDS_PW_OUT = f'{CALC_DIR}/bands_pw.out'
OUTPUT_PDF = f'{CALC_DIR}/bands_000.pdf'


def read_occupied_bands_from_pw_out(pw_out_path):
    """
    Read the 'number of Kohn-Sham states' from pw.out.
    This is the occupied-band count for a spin-unpolarized insulator.
    INVARIANT 2: never assume this from a per-element valence count.
    """
    with open(pw_out_path, 'r') as f:
        content = f.read()

    match = re.search(r'number of Kohn-Sham states\s*=\s*(\d+)', content)
    if match is None:
        raise ValueError(f"Could not find 'number of Kohn-Sham states' in {pw_out_path}")
    n_ks = int(match.group(1))

    match_e = re.search(r'number of electrons\s*=\s*([\d.]+)', content)
    if match_e is not None:
        n_elec = float(match_e.group(1))
        assert abs(n_ks - n_elec / 2) < 0.5, (
            f"Inconsistency: n_ks={n_ks}, n_elec/2={n_elec/2}"
        )
        print(f"  number of electrons = {n_elec}, number of Kohn-Sham states = {n_ks}")
    else:
        print(f"  number of Kohn-Sham states = {n_ks}")

    return n_ks


def read_band_labels(labels_path):
    """
    Read band_labels.dat: each line is 'LABEL INDEX'.
    Returns a list of (label, index) tuples.
    """
    labels = []
    with open(labels_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            label = parts[0]
            index = int(parts[1])
            labels.append((label, index))
    return labels


def read_bands_gnu(gnu_path):
    """
    Parse the .gnu file: one block per band, blocks separated by blank lines.
    Each block has n_kpts lines of (k_distance, energy).
    Returns (k_col, energies) where k_col is (n_kpts,) and energies is (n_kpts, n_bands).
    """
    blocks = []
    current_block = []
    with open(gnu_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_block:
                    blocks.append(current_block)
                    current_block = []
            else:
                parts = line.split()
                current_block.append([float(parts[0]), float(parts[1])])
    if current_block:
        blocks.append(current_block)

    n_bands = len(blocks)
    n_kpts = len(blocks[0])
    # All blocks must have the same number of lines
    for i, block in enumerate(blocks):
        assert len(block) == n_kpts, f"Block {i} has {len(block)} lines, expected {n_kpts}"

    k_col = np.array([b[0] for b in blocks[0]])
    energies = np.array([[b[j][1] for b in blocks] for j in range(n_kpts)])
    # energies shape: (n_kpts, n_bands)
    return k_col, energies


def read_nbnd_from_bands_in(bands_in_path):
    """Read nbnd from bands.in for sanity check."""
    with open(bands_in_path, 'r') as f:
        content = f.read()
    match = re.search(r'nbnd\s*=\s*(\d+)', content)
    if match is None:
        raise ValueError(f"Could not find 'nbnd' in {bands_in_path}")
    return int(match.group(1))


def read_nkpts_from_bands_pw_out(bands_pw_out_path):
    """Read 'number of k points' from bands_pw.out for sanity check."""
    with open(bands_pw_out_path, 'r') as f:
        content = f.read()
    match = re.search(r'number of k points\s*=\s*(\d+)', content)
    if match is None:
        raise ValueError(f"Could not find 'number of k points' in {bands_pw_out_path}")
    return int(match.group(1))


def main():
    print("Reading occupied band count from pw.out...")
    n_occupied = read_occupied_bands_from_pw_out(PW_OUT)
    print(f"  Occupied bands = {n_occupied}")

    print("Reading band labels...")
    labels = read_band_labels(BAND_LABELS)
    print(f"  {len(labels)} high-symmetry labels: {labels}")

    print("Reading bands data from .gnu file...")
    k_col, energies = read_bands_gnu(BANDS_GNU)
    n_kpts = len(k_col)
    n_bands = energies.shape[1]
    print(f"  {n_kpts} k-points, {n_bands} bands")
    print(f"  k range: {k_col.min():.4f} to {k_col.max():.4f}")

    # Sanity checks against bands.in and bands_pw.out
    nbnd_expected = read_nbnd_from_bands_in(BANDS_IN)
    assert n_bands == nbnd_expected, f"n_bands={n_bands} from .gnu file, but nbnd={nbnd_expected} in bands.in"
    print(f"  Sanity check: n_bands={n_bands} matches nbnd={nbnd_expected} in bands.in")

    nkpts_expected = read_nkpts_from_bands_pw_out(BANDS_PW_OUT)
    assert n_kpts == nkpts_expected, f"n_kpts={n_kpts} from .gnu file, but 'number of k points'={nkpts_expected} in bands_pw.out"
    print(f"  Sanity check: n_kpts={n_kpts} matches 'number of k points'={nkpts_expected} in bands_pw.out")

    # INVARIANT 2: VBM = max energy of the last occupied band (index n_occupied-1)
    # The bands are ordered by energy at each k-point, so the VBM is the max
    # of the (n_occupied-1)-th band across all k-points.
    assert n_occupied <= n_bands, f"n_occupied={n_occupied} > n_bands={n_bands}"
    vbm = np.max(energies[:, n_occupied - 1])
    # CBM = min energy of the first unoccupied band (index n_occupied)
    cbm = np.min(energies[:, n_occupied])
    band_gap = cbm - vbm
    print(f"  VBM = {vbm:.4f} eV")
    print(f"  CBM = {cbm:.4f} eV")
    print(f"  Band gap = {band_gap:.4f} eV")

    # Align energies to VBM: subtract VBM from all energies
    energies_vbm = energies - vbm

    # INVARIANT 4: tick positions must be in the same units as k_col.
    # band_labels.dat gives k-point INDICES, so map index -> k_col[index].
    tick_positions = [k_col[idx] for _, idx in labels]
    tick_labels = [lbl for lbl, _ in labels]

    # Assert that tick positions span approximately the same range as k_col
    assert abs(max(tick_positions) - k_col.max()) < 0.05 * k_col.max(), (
        f"INVARIANT 4 violation: max tick position {max(tick_positions)} "
        f"does not match k_col.max() {k_col.max()}"
    )
    assert abs(min(tick_positions) - k_col.min()) < 0.05 * k_col.max(), (
        f"INVARIANT 4 violation: min tick position {min(tick_positions)} "
        f"does not match k_col.min() {k_col.min()}"
    )
    print(f"  INVARIANT 4 check passed: tick positions span {min(tick_positions):.4f} to {max(tick_positions):.4f}, "
          f"k_col spans {k_col.min():.4f} to {k_col.max():.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for i_band in range(n_bands):
        ax.plot(k_col, energies_vbm[:, i_band], 'b-', linewidth=0.5)

    # Add high-symmetry tick marks
    for pos, lbl in zip(tick_positions, tick_labels):
        ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_xlabel(r'k-path (2$\pi$/a$_{lat}$)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title(r'Band structure of rutile TiO$_2$ (calculations/000)')
    ax.set_xlim(k_col.min(), k_col.max())
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, format='pdf')
    print(f"  Saved plot to {OUTPUT_PDF}")


if __name__ == '__main__':
    main()
