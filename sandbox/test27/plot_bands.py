#!/usr/bin/env python3
"""
Plot the band structure for structure 000 (unperturbed rutile TiO2).

Reads:
  - calculations/000/000.bands.dat (bands.x output: &plot header, then per k-point:
    3 k-coordinates in 2*pi/alat units, followed by nbnd energies in eV)
  - calculations/000/band_labels.dat (high-symmetry labels and their k-point indices)
  - calculations/000/pw.out (to get 'number of electrons' for VBM alignment)

Writes:
  - calculations/000/bands_000.pdf

Prints:
  - VBM, CBM, and band gap (in eV)

INVARIANT 2: occupied-band count read from pw.out, not assumed.
INVARIANT 4: tick positions and band data x-axis in the same units (cumulative distance).
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_bands_dat(filepath):
    """
    Parse the bands.x output file (filband format).
    
    Format:
      &plot nbnd=NN, nks=NNN /
      kx ky kz          (in 2*pi/alat units)
      e1 e2 ... e10      (10 energies per line, nbnd total per k-point)
      ...
    
    Returns:
        kpts: (nks, 3) array of k-coordinates in 2*pi/alat units
        energies: (nks, nbnd) array of band energies in eV
    """
    with open(filepath) as f:
        lines = f.readlines()
    
    # Parse header
    header = lines[0].strip()
    m = re.match(r'&plot\s+nbnd=\s*(\d+),\s*nks=\s*(\d+)\s*/', header)
    assert m, f"Could not parse header: {header}"
    nbnd = int(m.group(1))
    nks = int(m.group(2))
    
    kpts = np.zeros((nks, 3))
    energies = np.zeros((nks, nbnd))
    
    line_idx = 1
    for i in range(nks):
        # Read k-point coordinates
        k_line = lines[line_idx].strip()
        kpts[i] = np.array([float(x) for x in k_line.split()])
        line_idx += 1
        
        # Read energies (nbnd values, 10 per line)
        n_read = 0
        while n_read < nbnd:
            e_line = lines[line_idx].strip()
            vals = [float(x) for x in e_line.split()]
            n_this = min(len(vals), nbnd - n_read)
            energies[i, n_read:n_read + n_this] = vals[:n_this]
            n_read += n_this
            line_idx += 1
    
    return kpts, energies


def main():
    # --- Read the bands data ---
    kpts, energies = parse_bands_dat('calculations/000/000.bands.dat')
    nkpts = len(kpts)
    nbands = energies.shape[1]
    print(f"Bands data: {nkpts} k-points, {nbands} bands")
    
    # --- Compute cumulative k-path distance ---
    # The k-coordinates are in 2*pi/alat units. Compute the cumulative distance
    # along the path. At discontinuities (where the path jumps), the distance
    # increment is zero (we don't add the jump distance).
    # 
    # We identify discontinuities from band_labels.dat: labels containing '|'
    # mark the start of a new segment after a jump.
    
    # --- Read band labels ---
    labels = []
    with open('calculations/000/band_labels.dat') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                lbl, idx = parts[0], int(parts[1])
                labels.append((lbl, idx))
    
    print(f"Band labels: {labels}")
    
    # Identify discontinuity indices: k-point indices where a new segment starts
    # (i.e., where the label contains '|')
    discontinuity_indices = set()
    for lbl, idx in labels:
        if '|' in lbl:
            discontinuity_indices.add(idx)
    
    print(f"Discontinuity indices: {sorted(discontinuity_indices)}")
    
    # Compute cumulative distance
    # At a discontinuity, the distance increment from the previous point is 0
    # (we don't count the jump distance)
    k_dist = np.zeros(nkpts)
    for i in range(1, nkpts):
        if i in discontinuity_indices:
            # This is a jump: distance increment is 0
            k_dist[i] = k_dist[i - 1]
        else:
            # Normal increment: Euclidean distance in 2*pi/alat units
            dk = np.linalg.norm(kpts[i] - kpts[i - 1])
            k_dist[i] = k_dist[i - 1] + dk
    
    print(f"Cumulative distance range: {k_dist.min():.4f} to {k_dist.max():.4f}")
    
    # --- Read pw.out to get the number of electrons (INVARIANT 2) ---
    with open('calculations/000/pw.out') as f:
        pw_out = f.read()
    
    # Find 'number of electrons = XX.XX'
    m = re.search(r'number of electrons\s*=\s*([\d.]+)', pw_out)
    assert m, "Could not find 'number of electrons' in pw.out"
    n_electrons = float(m.group(1))
    print(f"Number of electrons from pw.out: {n_electrons}")
    
    # For spin-unpolarized non-metallic run: occupied bands = n_electrons / 2
    n_occ = int(n_electrons / 2)
    print(f"Occupied bands: {n_occ}")
    
    # INVARIANT 2: assert nbnd in the calculation is >= n_occ
    assert nbands >= n_occ, f"nbnd={nbands} < n_occ={n_occ} -- bands data is truncated!"
    
    # --- Find VBM and CBM ---
    # VBM = maximum energy among occupied bands (bands 0 to n_occ-1, 0-based)
    # CBM = minimum energy among unoccupied bands (bands n_occ to nbands-1)
    vbm = energies[:, :n_occ].max()
    cbm = energies[:, n_occ:].min()
    band_gap = cbm - vbm
    
    print(f"\nVBM = {vbm:.4f} eV")
    print(f"CBM = {cbm:.4f} eV")
    print(f"Band gap = {band_gap:.4f} eV")
    
    # --- Align energies to VBM ---
    energies_vbm = energies - vbm
    
    # --- Build tick positions from labels ---
    # Convert label indices to their corresponding k_dist values
    tick_positions = []
    tick_labels = []
    for lbl, idx in labels:
        if idx < nkpts:
            tick_positions.append(k_dist[idx])
            tick_labels.append(lbl)
    
    tick_positions = np.array(tick_positions)
    tick_labels = list(tick_labels)
    
    # INVARIANT 4: assert tick positions span approximately the same range as k_dist
    assert abs(tick_positions.max() - k_dist.max()) < 0.05 * k_dist.max(), \
        f"Tick positions max={tick_positions.max():.4f} vs k_dist max={k_dist.max():.4f} -- unit mismatch!"
    assert abs(tick_positions.min() - k_dist.min()) < 0.05 * k_dist.max(), \
        f"Tick positions min={tick_positions.min():.4f} vs k_dist min={k_dist.min():.4f} -- unit mismatch!"
    
    print(f"\nTick positions: {tick_positions}")
    print(f"Tick labels: {tick_labels}")
    
    # --- Plot the band structure ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each band, splitting at discontinuities
    # The discontinuity indices mark where a new segment starts
    # So the segments are: [0, disc1), [disc1, disc2), [disc2, disc3), ...
    boundaries = [0] + sorted(discontinuity_indices) + [nkpts]
    
    for band_idx in range(nbands):
        band_energies = energies_vbm[:, band_idx]
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            if end - start < 2:
                continue
            ax.plot(k_dist[start:end], band_energies[start:end], 'b-', linewidth=0.8)
    
    # Add tick marks at high-symmetry points
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, ha='center')
    
    # Add vertical lines at high-symmetry points
    for pos in tick_positions:
        ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Set y-axis range: +/-10 eV around VBM (which is now at 0)
    ax.set_ylim(-10, 10)
    ax.set_xlabel(r'k-point path (2$\pi$/a)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title(r'Rutile TiO$_2$ Band Structure (PBEsol, SCF)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
    print("\nSaved plot to calculations/000/bands_000.pdf")
    
    # --- Final summary ---
    print("\n=== Summary ===")
    print(f"VBM = {vbm:.4f} eV")
    print(f"CBM = {cbm:.4f} eV")
    print(f"Band gap = {band_gap:.4f} eV")
    print(f"Number of electrons: {n_electrons}")
    print(f"Occupied bands: {n_occ}")
    print(f"Total bands plotted: {nbands}")


if __name__ == '__main__':
    main()
