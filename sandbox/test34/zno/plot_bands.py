import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from chem_llm.kpath import plot_axis

# =============================================================================
# plot_bands.py
# Plots the band structure for calculations/000 (wurtzite ZnO, P6_3mc #186).
#
# Energy alignment (INVARIANT 2):
#   ZnO is a wide-gap insulator (occupations='fixed'), so we align to the VBM.
#   The occupied-band count is read from pw.out ('number of Kohn-Sham states'),
#   NOT assumed from a textbook valence count. The VBM is the highest occupied
#   band energy across all k-points.
#
# X-axis (INVARIANT 4):
#   plot_axis(kpath.json, bands.dat) returns x (path length), energies, ticks,
#   and segments. It also verifies that bands.x computed the same k-points as
#   kpath.json. We draw one line per band per segment (no line across a
#   discontinuity), a vertical line at every tick, and label ticks from the
#   ticks list (discontinuities already read 'A|L').
#
# Y-range: +/-10 eV around the VBM (semicore bands sit tens of eV below the
# valence manifold; we restrict the view to the relevant region).
# =============================================================================

# --- Read occupied-band count from pw.out (INVARIANT 2) ---
with open('calculations/000/pw.out') as f:
    pw_out = f.read()

# 'number of Kohn-Sham states' = occupied band count for spin-unpolarized run
m = re.search(r'number of Kohn-Sham states\s*=\s*(\d+)', pw_out)
assert m, "Could not find 'number of Kohn-Sham states' in pw.out"
n_occ = int(m.group(1))
print(f"Occupied bands (from pw.out): {n_occ}")

# Also read 'number of electrons' for cross-check
m_e = re.search(r'number of electrons\s*=\s*([\d.]+)', pw_out)
assert m_e, "Could not find 'number of electrons' in pw.out"
n_elec = float(m_e.group(1))
assert abs(n_elec - 2 * n_occ) < 1e-6, \
    f"Electron count mismatch: {n_elec} != 2*{n_occ}"
print(f"Number of electrons (from pw.out): {n_elec}")

# --- Get x-axis, energies, ticks, segments from plot_axis (INVARIANT 4) ---
axis = plot_axis('calculations/000/kpath.json', 'calculations/000/000.bands.dat')

x = axis['x']
energies = axis['energies']  # shape (n_kpts, n_bands), in eV
ticks = axis['ticks']        # list of dicts: {'x': float, 'label': str, 'indices': list}
segments = axis['segments']  # list of (start_idx, end_idx)

n_kpts = len(x)
n_bands = energies.shape[1]
print(f"k-points: {n_kpts}, bands: {n_bands}")
print(f"Ticks: {ticks}")
print(f"Segments: {segments}")

# --- Align to VBM ---
# The VBM is the maximum energy among the occupied bands (first n_occ bands)
# across all k-points.
occ_energies = energies[:, :n_occ]  # shape (n_kpts, n_occ)
vbm = occ_energies.max()
print(f"VBM (before alignment): {vbm:.6f} eV")

# Shift all energies so VBM = 0
e_shifted = energies - vbm

# --- Find CBM (lowest conduction band energy after alignment) ---
# Conduction bands are bands n_occ through n_bands-1
cb_energies = e_shifted[:, n_occ:]
cbm = cb_energies.min()
band_gap = cbm  # since VBM is at 0
print(f"CBM (after alignment): {cbm:.6f} eV")
print(f"Band gap: {band_gap:.6f} eV")

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))

# Draw each band as one line per segment (never across a discontinuity)
for band_idx in range(n_bands):
    for (start, end) in segments:
        seg_x = x[start:end + 1]
        seg_e = e_shifted[start:end + 1, band_idx]
        ax.plot(seg_x, seg_e, 'b-', linewidth=0.8)

# Draw vertical lines at every tick and label them
# ticks is a list of dicts with keys 'x', 'label', 'indices'
tick_positions = [t['x'] for t in ticks]
tick_labels = [t['label'] for t in ticks]

for t in ticks:
    ax.axvline(t['x'], color='k', linewidth=0.5, linestyle='-')

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)

# Set x-limits to the full path
ax.set_xlim(x[0], x[-1])

# Restrict y-axis to about +/-10 eV around VBM (at 0)
ax.set_ylim(-10, 10)

ax.set_xlabel('k-path')
ax.set_ylabel('Energy (eV) relative to VBM')
ax.set_title('ZnO wurtzite (P6_3mc) band structure\nPBEsol, ecutwfc=92 Ry, 8x8x4 k-mesh')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print("Saved calculations/000/bands_000.pdf")

# --- Print summary ---
print(f"\n=== Band structure summary for calculations/000 ===")
print(f"VBM: {vbm:.6f} eV (aligned to 0)")
print(f"CBM: {cbm:.6f} eV (relative to VBM)")
print(f"Band gap: {band_gap:.6f} eV")
print(f"Occupied bands: {n_occ}, Total bands plotted: {n_bands}")
