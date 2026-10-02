#!/usr/bin/env python3
"""
plot_bands.py -- Plot the band structure for calculations/000 (CsPbBr3, Pm-3m).

Reads:
  - calculations/000/000.bands.dat  (bands.x output: &plot header + k-points + energies in eV)
  - calculations/000/band_labels.dat (high-symmetry labels and their indices in the k-point list)
  - calculations/000/pw.out          (to read 'number of electrons' for VBM alignment)

Writes:
  - calculations/000/bands_000.pdf

Prints:
  - VBM, CBM, and band gap (in eV)

Key details:
  - Spin-orbit is ON (noncolin=.true., lspinorb=.true.), so each band holds ONE electron.
    Occupied bands = number of electrons = 82 (from pw.out).
    VBM = max energy of band index 81 (0-based), CBM = min energy of band index 82.
  - The k column in 000.bands.dat is cumulative path distance in 2*pi/alat units.
    Tick positions from band_labels.dat are indices into the k-point list; we must
    convert them to the same cumulative-distance units (INVARIANT 4).
  - Discontinuity at index 105 (X|R): no line drawn across the jump.
  - Y-axis restricted to +/-10 eV around VBM.
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CALC_DIR = 'calculations/000'
BANDS_DAT = f'{CALC_DIR}/000.bands.dat'
LABELS_DAT = f'{CALC_DIR}/band_labels.dat'
PW_OUT = f'{CALC_DIR}/pw.out'
OUTPUT_PDF = f'{CALC_DIR}/bands_000.pdf'


def read_number_of_electrons(pw_out_path):
    """Read 'number of electrons' from pw.out."""
    with open(pw_out_path) as f:
        for line in f:
            m = re.search(r'number of electrons\s*=\s*([\d.]+)', line)
            if m:
                return float(m.group(1))
    raise ValueError(f"Could not find 'number of electrons' in {pw_out_path}")


def parse_bands_dat(path):
    """
    Parse the bands.x output file (000.bands.dat).
    
    Format:
      &plot nbnd=100, nks=126 /
      kx ky kz
      e1 e2 ... e10
      e11 e12 ... e20
      ...
      (10 energies per line, nbnd/10 lines per k-point)
      (repeated for each k-point)
    
    Returns:
      k_values: (nks,) array of cumulative path distances (2*pi/alat)
      energies: (nks, nbnd) array of energies in eV
    """
    with open(path) as f:
        lines = f.readlines()
    
    # Parse header
    header = lines[0].strip()
    m_nbnd = re.search(r'nbnd=\s*(\d+)', header)
    m_nks = re.search(r'nks=\s*(\d+)', header)
    nbnd = int(m_nbnd.group(1))
    nks = int(m_nks.group(1))
    
    # Parse k-points and energies
    k_values = np.zeros(nks)
    energies = np.zeros((nks, nbnd))
    
    line_idx = 1
    for ik in range(nks):
        # Read k-point line
        k_line = lines[line_idx].strip()
        line_idx += 1
        k_parts = k_line.split()
        k_values[ik] = float(k_parts[0])  # cumulative distance (first column)
        
        # Read energies: nbnd values, 10 per line
        n_energy_lines = (nbnd + 9) // 10
        for iel in range(n_energy_lines):
            e_line = lines[line_idx].strip()
            line_idx += 1
            e_parts = e_line.split()
            for j, e in enumerate(e_parts):
                band_idx = iel * 10 + j
                if band_idx < nbnd:
                    energies[ik, band_idx] = float(e)
    
    return k_values, energies


def parse_band_labels(path):
    """
    Parse band_labels.dat: lines of 'label index'.
    Returns list of (label, index) tuples.
    """
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            label = parts[0]
            idx = int(parts[1])
            labels.append((label, idx))
    return labels


def main():
    # Read number of electrons from pw.out
    n_electrons = read_number_of_electrons(PW_OUT)
    print(f"Number of electrons (from pw.out): {n_electrons}")
    
    # With spin-orbit, each band holds 1 electron.
    # Occupied bands = n_electrons (integer)
    n_occ = int(round(n_electrons))
    print(f"Occupied bands (spin-orbit, 1 electron/band): {n_occ}")
    
    # Parse bands data
    k_values, energies = parse_bands_dat(BANDS_DAT)
    nks, nbnd = energies.shape
    print(f"Parsed {nks} k-points, {nbnd} bands")
    
    # INVARIANT 2: nbnd must be >= n_occ
    assert nbnd >= n_occ, f"nbnd={nbnd} < n_occ={n_occ}"
    
    # VBM: max energy of the highest occupied band (index n_occ-1, 0-based)
    vbm_band = energies[:, n_occ - 1]
    vbm = np.max(vbm_band)
    vbm_k_idx = np.argmax(vbm_band)
    
    # CBM: min energy of the lowest unoccupied band (index n_occ, 0-based)
    cbm_band = energies[:, n_occ]
    cbm = np.min(cbm_band)
    cbm_k_idx = np.argmin(cbm_band)
    
    band_gap = cbm - vbm
    print(f"VBM = {vbm:.4f} eV (at k-point index {vbm_k_idx})")
    print(f"CBM = {cbm:.4f} eV (at k-point index {cbm_k_idx})")
    print(f"Band gap = {band_gap:.4f} eV")
    
    # Align energies to VBM
    energies_aligned = energies - vbm
    
    # Parse band labels
    labels = parse_band_labels(LABELS_DAT)
    print(f"Band labels: {labels}")
    
    # Convert label indices to cumulative k-distance (INVARIANT 4)
    # The k_values array is cumulative path distance in 2*pi/alat units.
    # Each label's index points into the k-point list, so the tick position
    # is k_values[index].
    tick_positions = []
    tick_labels = []
    for label, idx in labels:
        assert idx < nks, f"Label {label} at index {idx} out of range (nks={nks})"
        tick_positions.append(k_values[idx])
        tick_labels.append(label)
    
    tick_positions = np.array(tick_positions)
    
    # INVARIANT 4 CHECK: tick positions must span approximately the same range
    # as the k data
    assert abs(tick_positions.max() - k_values.max()) < 0.05 * k_values.max(), \
        f"Tick positions max={tick_positions.max():.4f} vs k data max={k_values.max():.4f}"
    assert abs(tick_positions.min() - k_values.min()) < 0.05 * max(k_values.max(), 1e-10), \
        f"Tick positions min={tick_positions.min():.4f} vs k data min={k_values.min():.4f}"
    
    # Identify discontinuities: where the label contains '|'
    # These are genuine path jumps -- no line should be drawn across them.
    discontinuity_indices = []
    for label, idx in labels:
        if '|' in label:
            discontinuity_indices.append(idx)
    
    print(f"Discontinuities at k-point indices: {discontinuity_indices}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bands, splitting at discontinuities
    # We need to break the line at each discontinuity index.
    # Build segments: from index 0 to first discontinuity, then from discontinuity+1 to next, etc.
    
    # Sort discontinuity indices
    disc_sorted = sorted(discontinuity_indices)
    
    # Build plot segments
    segments = []
    start = 0
    for d in disc_sorted:
        # Segment from start to d (inclusive of d, which is the 'from' side of the jump)
        segments.append((start, d))
        start = d + 1  # next segment starts after the jump
    # Final segment
    segments.append((start, nks - 1))
    
    for (seg_start, seg_end) in segments:
        k_seg = k_values[seg_start:seg_end + 1]
        e_seg = energies_aligned[seg_start:seg_end + 1, :]
        # Plot each band in this segment
        for ib in range(nbnd):
            ax.plot(k_seg, e_seg[:, ib], 'b-', linewidth=0.5)
    
    # Set x-axis ticks at high-symmetry points
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, ha='center')
    
    # Y-axis: +/-10 eV around VBM (which is now at 0)
    ax.set_ylim(-10, 10)
    ax.set_ylabel('Energy (eV) relative to VBM')
    ax.set_xlabel('k-path (2$\pi$/a units)')
    ax.set_title('CsPbBr3 band structure (PBEsol + spin-orbit)')
    ax.axhline(0, color='k', linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, dpi=150)
    print(f"Saved plot to {OUTPUT_PDF}")


if __name__ == '__main__':
    main()
