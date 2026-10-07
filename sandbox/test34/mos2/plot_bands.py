import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from chem_llm.kpath import plot_axis

# =============================================================================
# STEP 7: plot_bands.py for calculations/000
# =============================================================================
# Reads the band structure from 000.bands.dat (produced by bands.x),
# aligns energies to the VBM using the occupied-band count from pw.out,
# and plots the band structure with proper x-axis from kpath.json.
# =============================================================================

# --- Read the number of Kohn-Sham states (occupied bands) from pw.out ---
# INVARIANT 2: the occupied-band count is a property of the pseudopotentials
# and the SCF run, not a guess. Read it from pw.out directly.
occupied_bands = None
with open('calculations/000/pw.out') as f:
    for line in f:
        if 'number of Kohn-Sham states' in line:
            occupied_bands = int(line.split('=')[1].strip())
            break

assert occupied_bands is not None, 'Could not find "number of Kohn-Sham states" in pw.out'
print(f'Occupied bands from pw.out: {occupied_bands}')

# --- Get the x-axis, energies, ticks, and segments from plot_axis ---
# INVARIANT 4: plot_axis checks that bands.x computed the same k-points as kpath.json
axis = plot_axis('calculations/000/kpath.json', 'calculations/000/000.bands.dat')

x = axis['x']           # path length (in A^-1 or similar units)
energies = axis['energies']  # shape (n_kpts, n_bands), in eV
ticks = axis['ticks']   # list of dicts with 'x', 'label', 'indices'
segments = axis['segments']  # list of (start_idx, end_idx) for each continuous segment

n_kpts = len(x)
n_bands = energies.shape[1]
print(f'k-points: {n_kpts}, bands: {n_bands}')
print(f'Ticks: {ticks}')
print(f'Segments: {segments}')

# --- Align energies to the VBM ---
# The VBM is the maximum energy among the occupied bands (bands 0 to occupied_bands-1).
# For a semiconductor, the VBM is the highest occupied state.
vbm = np.max(energies[:, :occupied_bands])
print(f'VBM (before alignment): {vbm:.4f} eV')

# Shift all energies so that VBM = 0
e = energies - vbm

# --- Find the CBM (minimum of conduction bands) ---
cbm = np.min(e[:, occupied_bands:])
band_gap = cbm - 0.0  # VBM is at 0
print(f'CBM: {cbm:.4f} eV')
print(f'Band gap: {band_gap:.4f} eV')

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for band_idx in range(n_bands):
    for seg_start, seg_end in segments:
        seg_x = x[seg_start:seg_end + 1]
        seg_e = e[seg_start:seg_end + 1, band_idx]
        ax.plot(seg_x, seg_e, 'b-', linewidth=0.8)

# Draw vertical lines at every tick
tick_xs = [t['x'] for t in ticks]
tick_labels = [t['label'] for t in ticks]
for tick_x in tick_xs:
    ax.axvline(x=tick_x, color='gray', linewidth=0.5, linestyle='--')

# Set x-axis ticks and labels
ax.set_xticks(tick_xs)
ax.set_xticklabels(tick_labels)

# Set x-limits to the full path
ax.set_xlim(x[0], x[-1])

# Restrict y-axis to about +/-10 eV around VBM (at 0)
ax.set_ylim(-10, 10)

# Labels
ax.set_xlabel('k-point path')
ax.set_ylabel('Energy (eV)')
ax.set_title(f'2H-MoS2 band structure (PBEsol, VBM-aligned)\nBand gap = {band_gap:.3f} eV')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print('Saved calculations/000/bands_000.pdf')

print(f'\nSummary:')
print(f'  VBM: {vbm:.4f} eV (aligned to 0)')
print(f'  CBM: {cbm:.4f} eV')
print(f'  Band gap: {band_gap:.4f} eV')
