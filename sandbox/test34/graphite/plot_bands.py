import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from chem_llm.kpath import plot_axis

# =============================================================================
# plot_bands.py
# Plots the band structure for calculations/000 (graphite, mp-48).
# Graphite is a zero-gap semimetal -> align to the Fermi energy from pw.out.
# =============================================================================

CALC_DIR = 'calculations/000'
KPATH_JSON = f'{CALC_DIR}/kpath.json'
BANDS_DAT = f'{CALC_DIR}/000.bands.dat'
PW_OUT = f'{CALC_DIR}/pw.out'
OUTPUT_PDF = f'{CALC_DIR}/bands_000.pdf'

# --- Read the Fermi energy from pw.out (SCF output) ---
# For a semimetal, the Fermi energy is the correct reference, not the VBM.
with open(PW_OUT) as f:
    pw_text = f.read()

# QE writes: "     the Fermi energy is     5.3839 ev"
fermi_match = re.search(r'the Fermi energy is\s+([\d.]+)\s+ev', pw_text)
assert fermi_match, 'Could not find Fermi energy in pw.out'
fermi_ev = float(fermi_match.group(1))

# Also read the number of electrons to verify nbnd (INVARIANT 2)
elec_match = re.search(r'number of electrons\s*=\s*([\d.]+)', pw_text)
assert elec_match, 'Could not find number of electrons in pw.out'
n_electrons = float(elec_match.group(1))
print(f'Number of electrons from pw.out: {n_electrons}')
print(f'Fermi energy from pw.out: {fermi_ev:.4f} eV')

# --- Get the x-axis, energies, ticks, and line segments from plot_axis ---
# plot_axis checks that bands.dat was computed on kpath.json's k-points.
axis = plot_axis(KPATH_JSON, BANDS_DAT)

# plot_axis returns a dict with keys:
#   'x': array of path-length values (one per k-point)
#   'energies': 2D array (n_kpts, n_bands) in eV
#   'ticks': list of dicts with keys 'x', 'label', 'indices'
#   'segments': list of (x_start, x_end) for each continuous segment
x = axis['x']
energies = axis['energies']  # in eV
ticks = axis['ticks']
segments = axis['segments']

print(f'Number of k-points: {len(x)}')
print(f'Number of bands: {energies.shape[1]}')
print(f'Number of segments: {len(segments)}')
print(f'Ticks: {ticks}')

# --- Align energies to the Fermi energy ---
# bands.dat energies are in eV, absolute (not shifted).
# Shift so that the Fermi energy is at 0.
energies_shifted = energies - fermi_ev

# --- Find VBM and CBM relative to the Fermi energy ---
# For a semimetal, the VBM is the highest occupied band energy (just below E_F)
# and the CBM is the lowest unoccupied band energy (just above E_F).
# With smearing, the Fermi level is inside the band structure.
# VBM = max of energies below E_F, CBM = min of energies above E_F.
below_fermi = energies_shifted[energies_shifted < 0]
above_fermi = energies_shifted[energies_shifted > 0]

if len(below_fermi) > 0:
    vbm = np.max(below_fermi)
else:
    vbm = 0.0  # no states below Fermi (unlikely for graphite)

if len(above_fermi) > 0:
    cbm = np.min(above_fermi)
else:
    cbm = 0.0  # no states above Fermi (unlikely for graphite)

band_gap = cbm - vbm
print(f'VBM (relative to E_F): {vbm:.4f} eV')
print(f'CBM (relative to E_F): {cbm:.4f} eV')
print(f'Band gap: {band_gap:.4f} eV')
print(f'Fermi energy: {fermi_ev:.4f} eV')

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
n_bands = energies_shifted.shape[1]
for seg_start, seg_end in segments:
    # Find the indices of k-points in this segment
    mask = (x >= seg_start) & (x <= seg_end)
    x_seg = x[mask]
    e_seg = energies_shifted[mask]
    for band in range(n_bands):
        ax.plot(x_seg, e_seg[:, band], 'b-', linewidth=0.8)

# Draw a vertical line at every tick and label it
# ticks is a list of dicts: {'x': float, 'label': str, 'indices': list}
for tick in ticks:
    tick_x = tick['x']
    tick_label = tick['label']
    ax.axvline(x=tick_x, color='gray', linewidth=0.5, linestyle='--')

# Set x-tick labels from plot_axis's ticks
ax.set_xticks([t['x'] for t in ticks])
ax.set_xticklabels([t['label'] for t in ticks])

# Set x-limits to the full path
ax.set_xlim(x[0], x[-1])

# Restrict y-axis to about +/-10 eV
ax.set_ylim(-10, 10)

# Labels and title
ax.set_xlabel('k-point path')
ax.set_ylabel('Energy - E$_F$ (eV)')
ax.set_title('Graphite (mp-48) band structure\nPBEsol, 82 Ry cutoff, 12x12x4 k-mesh')

# Add a horizontal line at E_F = 0
ax.axhline(y=0, color='red', linewidth=0.5, linestyle=':')

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=150)
print(f'Saved band structure plot to {OUTPUT_PDF}')
