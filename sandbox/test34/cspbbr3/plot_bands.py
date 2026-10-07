#!/usr/bin/env python3
"""
plot_bands.py: Parse the bands.x output for calculations/000 and plot the band structure.

Reads:
  - calculations/000/000.bands.dat (primary filband file from bands.x)
  - calculations/000/band_labels.dat (high-symmetry labels and indices)
  - calculations/000/pw.out (for 'number of electrons' to determine VBM)

The 000.bands.dat format:
  Header: &plot nbnd=100, nks=116 /
  Then 116 k-point blocks, each with:
    - 3 Cartesian k-coordinates (kx ky kz in 2*pi/alat)
    - 100 energies in eV (10 per line)

We compute the cumulative path distance from consecutive k-points.
At a discontinuity (index 96, X->R jump), we add NO distance across the jump.

With spin-orbit, each band holds ONE electron.
Total valence electrons = 82 (from pw.out), so occupied bands = 82.
VBM = band 82 (index 81 in 0-based), CBM = band 83 (index 82).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import os

CALC_DIR = 'calculations/000'
BANDS_FILE = os.path.join(CALC_DIR, '000.bands.dat')
LABELS_FILE = os.path.join(CALC_DIR, 'band_labels.dat')
PW_OUT = os.path.join(CALC_DIR, 'pw.out')
OUTPUT_PDF = os.path.join(CALC_DIR, 'bands_000.pdf')


def parse_bands_dat(filepath):
    """
    Parse the bands.x filband output file.
    
    Returns:
      kpoints: numpy array (n_kpts, 3) -- Cartesian k-coordinates in 2*pi/alat
      energies: numpy array (n_kpts, nbnd) -- energies in eV
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    header = lines[0].strip()
    m = re.match(r'&plot\s+nbnd=\s*(\d+),\s*nks=\s*(\d+)\s*/', header)
    assert m, f"Could not parse header: {header}"
    nbnd = int(m.group(1))
    nks = int(m.group(2))
    print(f"Header: nbnd={nbnd}, nks={nks}")
    
    kpoints = []
    energies = []
    
    # Skip header line
    i = 1
    while i < len(lines) and len(kpoints) < nks:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        # Check if this line has 3 floats (k-point coordinates)
        parts = line.split()
        if len(parts) == 3:
            try:
                kx, ky, kz = float(parts[0]), float(parts[1]), float(parts[2])
                kpoints.append([kx, ky, kz])
                i += 1
                
                # Now read nbnd energies (10 per line)
                band_energies = []
                while len(band_energies) < nbnd and i < len(lines):
                    eline = lines[i].strip()
                    if not eline:
                        i += 1
                        continue
                    eparts = eline.split()
                    # Check if this line is a k-point coordinate line (3 floats)
                    # or an energy line (10 floats)
                    if len(eparts) == 3:
                        # This is the next k-point, stop reading energies
                        break
                    band_energies.extend(float(x) for x in eparts)
                    i += 1
                
                assert len(band_energies) == nbnd, \
                    f"Expected {nbnd} energies, got {len(band_energies)}"
                energies.append(band_energies)
            except ValueError:
                # Not a k-point line, skip
                i += 1
        else:
            i += 1
    
    assert len(kpoints) == nks, f"Expected {nks} k-points, got {len(kpoints)}"
    assert len(energies) == nks, f"Expected {nks} energy blocks, got {len(energies)}"
    
    return np.array(kpoints), np.array(energies)


def compute_cumulative_distance(kpoints, discontinuity_indices):
    """
    Compute cumulative path distance from consecutive k-points.
    At a discontinuity, add NO distance across the jump.
    
    Returns:
      x: numpy array (n_kpts,) -- cumulative path distance in 2*pi/alat
    """
    n = len(kpoints)
    x = np.zeros(n)
    
    for i in range(1, n):
        if i in discontinuity_indices:
            # Discontinuity: no distance added across the jump
            x[i] = x[i - 1]
        else:
            dk = np.linalg.norm(kpoints[i] - kpoints[i - 1])
            x[i] = x[i - 1] + dk
    
    # INVARIANT 4: x values must never decrease
    assert np.all(np.diff(x) >= -1e-10), "Cumulative distance decreased somewhere"
    
    return x


def read_band_labels(filepath):
    """
    Read band_labels.dat: each line is 'index label'.
    Returns dict: index -> label
    """
    labels = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            idx = int(parts[0])
            label = parts[1]
            labels[idx] = label
    return labels


def read_num_electrons(pw_out_path):
    """
    Read 'number of electrons' from pw.out.
    The line format is: '     number of electrons       =        82.00'
    Use a regex to extract the number after 'number of electrons', allowing
    an optional '=' sign in between.
    """
    with open(pw_out_path, 'r') as f:
        for line in f:
            if 'number of electrons' in line.lower():
                # Match: 'number of electrons' followed by optional '=' and a float
                m = re.search(r'number of electrons\s*=?\s*([\d.]+)', line, re.IGNORECASE)
                if m:
                    return float(m.group(1))
    raise ValueError("Could not find 'number of electrons' in pw.out")


def main():
    # Parse bands data
    kpoints, energies = parse_bands_dat(BANDS_FILE)
    n_kpts, nbnd = energies.shape
    print(f"Parsed {n_kpts} k-points, {nbnd} bands")
    
    # Read band labels
    labels = read_band_labels(LABELS_FILE)
    print(f"Band labels: {labels}")
    
    # Determine discontinuity indices from labels
    # A discontinuity is where a label contains '|'
    discontinuity_indices = set()
    for idx, label in labels.items():
        if '|' in label:
            discontinuity_indices.add(idx)
    print(f"Discontinuity indices: {discontinuity_indices}")
    
    # Compute cumulative path distance
    x = compute_cumulative_distance(kpoints, discontinuity_indices)
    print(f"Cumulative distance range: {x[0]:.4f} to {x[-1]:.4f} (2*pi/alat)")
    
    # Read number of electrons from pw.out
    n_electrons = read_num_electrons(PW_OUT)
    print(f"Number of electrons from pw.out: {n_electrons}")
    
    # With spin-orbit, each band holds 1 electron
    # Occupied bands = n_electrons (should be integer)
    n_occ = int(round(n_electrons))
    assert abs(n_electrons - n_occ) < 1e-6, \
        f"Number of electrons {n_electrons} is not an integer (expected for spin-orbit)"
    print(f"Occupied bands (spin-orbit): {n_occ}")
    
    # INVARIANT 2: nbnd must be >= n_occ
    assert nbnd >= n_occ, f"nbnd={nbnd} < n_occ={n_occ}"
    
    # VBM = highest occupied band = band n_occ (index n_occ-1 in 0-based)
    # CBM = lowest unoccupied band = band n_occ+1 (index n_occ in 0-based)
    vbm = np.max(energies[:, n_occ - 1])
    cbm = np.min(energies[:, n_occ])
    gap = cbm - vbm
    print(f"VBM = {vbm:.4f} eV")
    print(f"CBM = {cbm:.4f} eV")
    print(f"Band gap = {gap:.4f} eV")
    
    # Align energies to VBM
    energies_vbm = energies - vbm
    
    # INVARIANT 4: tick positions must span the same range as the band data's x column
    tick_indices = sorted(labels.keys())
    tick_positions = x[tick_indices]
    assert abs(tick_positions[-1] - x[-1]) < 0.05 * x[-1], \
        f"Tick positions max {tick_positions[-1]:.4f} vs x max {x[-1]:.4f} -- unit mismatch"
    print(f"Tick positions span: {tick_positions[0]:.4f} to {tick_positions[-1]:.4f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each band, splitting at discontinuities
    for band_idx in range(nbnd):
        e = energies_vbm[:, band_idx]
        # Only plot bands within +/-10 eV of VBM
        if np.max(np.abs(e)) < 10.0 or np.min(e) < 10.0:
            # Split at discontinuities
            segments = []
            current_seg = [0]
            for i in range(1, n_kpts):
                if i in discontinuity_indices:
                    segments.append(current_seg)
                    current_seg = [i]
                else:
                    current_seg.append(i)
            segments.append(current_seg)
            
            for seg in segments:
                ax.plot(x[seg], e[seg], 'b-', linewidth=0.5)
    
    # Add high-symmetry ticks
    for idx in tick_indices:
        label = labels[idx]
        ax.axvline(x[idx], color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([labels[i] for i in tick_indices], rotation=0)
    
    # Y-axis: +/-10 eV around VBM
    ax.set_ylim(-10, 10)
    ax.set_xlabel(r'k-path (2$\pi$/a)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title(f'CsPbBr3 band structure (PBEsol + SOC)\nGap = {gap:.3f} eV')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {OUTPUT_PDF}")
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"VBM = {vbm:.4f} eV")
    print(f"CBM = {cbm:.4f} eV")
    print(f"Band gap = {gap:.4f} eV")
    print(f"Number of electrons: {n_electrons}")
    print(f"Occupied bands: {n_occ}")
    print(f"nbnd: {nbnd}")
    print(f"K-points: {n_kpts}")
    print(f"Discontinuities at indices: {sorted(discontinuity_indices)}")


if __name__ == '__main__':
    main()
