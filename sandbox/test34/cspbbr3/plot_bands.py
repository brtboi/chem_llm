import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from chem_llm.kpath import plot_axis

# =============================================================================
# plot_bands.py
# Plots the band structure for calculations/000 (CsPbBr3, spin-orbit on).
# - x-axis, ticks, segments from chem_llm.kpath.plot_axis (INVARIANT 4)
# - VBM alignment using occupied-band count read from pw.out (INVARIANT 2)
# - Each band drawn as one line per segment (no line across discontinuities)
# - Vertical lines at every tick, labels from plot_axis ticks
# - y-axis restricted to about +/-10 eV around VBM
# =============================================================================

CALC_DIR = 'calculations/000'
KPATH_JSON = f'{CALC_DIR}/kpath.json'
BANDS_DAT = f'{CALC_DIR}/000.bands.dat'
PW_OUT = f'{CALC_DIR}/pw.out'
OUTPUT_PDF = f'{CALC_DIR}/bands_000.pdf'

# --- Read the number of electrons from pw.out (INVARIANT 2) ---
# With spin-orbit (noncolin=.true.), each band holds ONE electron,
# so occupied bands = number of electrons.
with open(PW_OUT) as f:
    pw_text = f.read()

m = re.search(r'number of electrons\s*=\s*([\d.]+)', pw_text)
assert m, "Could not find 'number of electrons' in pw.out"
n_electrons = float(m.group(1))
print(f"Number of electrons from pw.out: {n_electrons}")

# With spin-orbit, each band holds 1 electron, so occupied bands = n_electrons
n_occ = int(round(n_electrons))
print(f"Occupied bands (spin-orbit): {n_occ}")

# --- Get x-axis, energies, ticks, segments from plot_axis (INVARIANT 4) ---
# plot_axis checks that bands.dat was computed on the k-points in kpath.json
axis = plot_axis(KPATH_JSON, BANDS_DAT)

# plot_axis returns a dict with keys: x, energies, ticks, segments
# x: array of path-length values (one per k-point)
# energies: 2D array (n_kpts, n_bands) in eV
# ticks: list of dicts with keys 'x', 'label', 'indices'
# segments: list of (start_idx, end_idx) index pairs for each continuous segment
x = axis['x']
energies = axis['energies']
ticks = axis['ticks']
segments = axis['segments']

print(f"Number of k-points: {len(x)}")
print(f"Number of bands: {energies.shape[1]}")
print(f"Number of segments: {len(segments)}")
print(f"Ticks: {ticks}")

# --- Align energies to VBM ---
# VBM = maximum energy among occupied bands (bands 0 to n_occ-1)
# With spin-orbit, band index 0 is the lowest, band index n_occ-1 is the VBM
occ_energies = energies[:, :n_occ]
vbm = np.max(occ_energies)
print(f"VBM (raw): {vbm:.6f} eV")

# Shift all energies so VBM = 0
e_shifted = energies - vbm

# --- Find CBM (minimum of conduction bands) ---
cond_energies = e_shifted[:, n_occ:]
cbm = np.min(cond_energies)
band_gap = cbm  # since VBM is at 0
print(f"CBM (relative to VBM): {cbm:.6f} eV")
print(f"Band gap: {band_gap:.6f} eV")

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for band_idx in range(energies.shape[1]):
    for (start, end) in segments:
        seg_x = x[start:end + 1]
        seg_e = e_shifted[start:end + 1, band_idx]
        ax.plot(seg_x, seg_e, 'b-', linewidth=0.8)

# Draw vertical lines at every tick and set tick labels
# ticks are dicts with keys 'x', 'label', 'indices'
tick_xs = [t['x'] for t in ticks]
tick_labels = [t['label'] for t in ticks]
for tick_x in tick_xs:
    ax.axvline(x=tick_x, color='k', linewidth=0.5, alpha=0.5)
ax.set_xticks(tick_xs)
ax.set_xticklabels(tick_labels)

# Set x-limits to the full path
ax.set_xlim(x[0], x[-1])

# Restrict y-axis to about +/-10 eV around VBM
ax.set_ylim(-10, 10)

ax.set_xlabel('k-path')
ax.set_ylabel('Energy (eV) relative to VBM')
ax.set_title(f'CsPbBr3 band structure (spin-orbit, PBEsol)\nVBM=0, CBM={cbm:.3f} eV, gap={band_gap:.3f} eV')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=150)
print(f"Saved plot to {OUTPUT_PDF}")
print(f"\nSummary:")
print(f"  VBM: 0.000 eV (by definition)")
print(f"  CBM: {cbm:.4f} eV")
print(f"  Band gap: {band_gap:.4f} eV")
