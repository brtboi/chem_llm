import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# =============================================================================
# plot_bands.py
# Plots the band structure for calculations/000 (rutile TiO2, unperturbed).
# Energies are aligned to the VBM (TiO2 is a semiconductor, not a metal).
# =============================================================================

KPATH_JSON = 'calculations/000/kpath.json'
BANDS_DAT = 'calculations/000/000.bands.dat'
PW_OUT = 'calculations/000/pw.out'
OUTPUT_PDF = 'calculations/000/bands_000.pdf'

# Get x-axis, energies, ticks, and line segments from plot_axis (INVARIANT 4)
# plot_axis checks that bands.dat was computed on the same k-points as kpath.json
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']          # path length (x-axis)
E = ax['energies']   # shape (n_kpts, n_bands), in eV
segments = ax['segments']       # slice objects, one per continuous segment
tick_x = ax['tick_x']           # list of x positions for high-symmetry points
tick_labels = ax['tick_labels'] # list of labels (e.g. 'GAMMA', 'X|L', etc.)

# Read SCF output to get n_occupied and Fermi energy (INVARIANT 2)
pw = read_pw_out(PW_OUT)
n_occupied = pw['n_occupied']
print(f'Number of occupied bands: {n_occupied}')
print(f'Fermi energy: {pw["fermi_energy_eV"]} eV')

# Align to VBM (TiO2 is a semiconductor, not a metal)
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f'VBM: {vbm:.4f} eV')
print(f'CBM: {cbm:.4f} eV')
print(f'Band gap: {gap:.4f} eV')

# Plot each band as one line per segment (never across a discontinuity)
fig, plt_ax = plt.subplots(figsize=(10, 6))
for s in segments:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], E[s, b] - vbm, 'k-', lw=1)

# Draw a vertical line at every tick
for t in tick_x:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Label the ticks
plt_ax.set_xticks(tick_x)
plt_ax.set_xticklabels(tick_labels)

# Set axis limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)  # restrict y-axis to about +/-10 eV around VBM
plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title('Rutile TiO$_2$ band structure (PBEsol, VBM-aligned)')

plt.tight_layout()
plt.savefig(OUTPUT_PDF)
print(f'Saved plot to {OUTPUT_PDF}')
