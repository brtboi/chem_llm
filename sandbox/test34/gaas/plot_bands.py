"""
plot_bands.py — Plot the band structure for calculations/000.

Reads the band energies from 000.bands.dat (produced by bands.x),
aligns them to the VBM (GaAs is a semiconductor), and produces a
publication-quality PDF plot.

Key steps:
  1. Read 'number of electrons' from pw.out to get the occupied-band count.
  2. Use chem_llm.kpath.plot_axis for x-axis, ticks, and line segments.
  3. Align energies to VBM: shift so the highest occupied band is at 0 eV.
  4. Plot each band as one line per segment (no line across discontinuities).
  5. Draw vertical lines at every tick, label ticks from plot_axis.
  6. Save to calculations/000/bands_000.pdf.
  7. Print VBM, CBM, and band gap.
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
from chem_llm.kpath import plot_axis

# --- Step 1: Read occupied-band count from pw.out ---
# INVARIANT 2: the occupied-band count is a property of the pseudopotentials,
# not the element. Read it from pw.out, not from memory.
with open('calculations/000/pw.out') as f:
    pw_out = f.read()

# Find 'number of electrons' line
m = re.search(r'number of electrons\s*=\s*([\d.]+)', pw_out)
assert m, 'Could not find "number of electrons" in pw.out'
n_electrons = float(m.group(1))
print(f'Number of electrons from pw.out: {n_electrons}')

# For spin-unpolarized non-metallic run: occupied bands = n_electrons / 2
n_occ = int(n_electrons / 2)
print(f'Occupied bands (n_electrons/2): {n_occ}')

# INVARIANT 2 check: nbnd in bands.in must be >= n_occ
with open('calculations/000/bands.in') as f:
    bands_in = f.read()
m_nbnd = re.search(r'nbnd\s*=\s*(\d+)', bands_in)
assert m_nbnd, 'Could not find nbnd in bands.in'
nbnd = int(m_nbnd.group(1))
assert nbnd >= n_occ, f'nbnd={nbnd} < occupied bands={n_occ}'
print(f'nbnd from bands.in: {nbnd} (>= {n_occ}, OK)')

# --- Step 2: Get x-axis, energies, ticks, segments from plot_axis ---
# INVARIANT 4: plot_axis checks that bands.x computed the same k-points
# as kpath.json, and returns x (path length), ticks, and segments.
axis = plot_axis('calculations/000/kpath.json', 'calculations/000/000.bands.dat')

x = axis['x']           # path length (array of shape (n_kpoints,))
energies = axis['energies']  # shape (n_kpoints, nbnd), in eV
ticks = axis['ticks']   # list of dicts with 'x', 'label', 'indices'
segments = axis['segments']  # list of (start_idx, end_idx) index pairs

print(f'k-points: {len(x)}, bands: {energies.shape[1]}')
print(f'Ticks: {ticks}')
print(f'Segments: {segments}')

# --- Step 3: Align energies to VBM ---
# The VBM is the maximum energy among the occupied bands (first n_occ bands).
# For a semiconductor, the VBM is at the top of the valence manifold.
vbm = np.max(energies[:, :n_occ])
print(f'VBM (before alignment): {vbm:.4f} eV')

# Shift all energies so VBM = 0
e_shifted = energies - vbm

# CBM is the minimum energy among the unoccupied bands (bands n_occ onward)
cbm = np.min(e_shifted[:, n_occ:])
print(f'CBM (after alignment): {cbm:.4f} eV')
print(f'Band gap: {cbm:.4f} eV')

# --- Step 4-6: Plot ---
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for band_idx in range(e_shifted.shape[1]):
    for seg_start, seg_end in segments:
        # Slice the segment (inclusive of both endpoints)
        seg_x = x[seg_start:seg_end + 1]
        seg_e = e_shifted[seg_start:seg_end + 1, band_idx]
        ax.plot(seg_x, seg_e, 'b-', linewidth=0.8, alpha=0.8)

# Draw vertical lines at every tick and set tick labels
for tick in ticks:
    ax.axvline(x=tick['x'], color='k', linewidth=0.5, alpha=0.5)

ax.set_xticks([t['x'] for t in ticks])
ax.set_xticklabels([t['label'] for t in ticks])

# Set x-limits to the full path
ax.set_xlim(x[0], x[-1])

# Restrict y-axis to about +/-10 eV (INVARIANT 4)
ax.set_ylim(-10, 10)

ax.set_xlabel('k-point path')
ax.set_ylabel('Energy (eV) relative to VBM')
ax.set_title('GaAs band structure (PBEsol, 2-atom primitive cell)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print('Saved calculations/000/bands_000.pdf')

# --- Step 7: Print summary ---
print(f'\nSummary:')
print(f'  VBM: {vbm:.4f} eV (aligned to 0)')
print(f'  CBM: {cbm:.4f} eV')
print(f'  Band gap: {cbm:.4f} eV')
