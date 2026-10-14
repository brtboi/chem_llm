import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# =============================================================================
# plot_bands.py
# Plots the band structure for calculations/000 (fcc Al, metal).
# Aligns to the Fermi energy (Al is a metal, band_gap_eV=0.0).
# =============================================================================

KPATH_JSON = 'calculations/000/kpath.json'
BANDS_DAT = 'calculations/000/000.bands.dat'
PW_OUT = 'calculations/000/pw.out'
OUTPUT_PDF = 'calculations/000/bands_000.pdf'

# --- Get x-axis, energies, ticks, and segments from plot_axis (INVARIANT 4) ---
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']          # path length (arbitrary units)
E = ax['energies']   # shape (n_kpts, n_bands), in eV
segments = ax['segments']
tick_x = ax['tick_x']
tick_labels = ax['tick_labels']

print(f"Bands shape: {E.shape}")
print(f"Number of segments: {len(segments)}")
print(f"Ticks: {list(zip(tick_x, tick_labels))}")

# --- Read pw.out to get n_occupied and Fermi energy (INVARIANT 2) ---
pw = read_pw_out(PW_OUT)
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']
print(f"n_occupied = {n_occupied}")
print(f"Fermi energy = {fermi_eV:.4f} eV")

# Al is a metal -> align to Fermi energy
zero = fermi_eV

# --- Plot bands: one line per segment, never across a discontinuity ---
fig, ax_plot = plt.subplots(figsize=(10, 6))

for s in segments:
    for b in range(E.shape[1]):
        ax_plot.plot(x[s], E[s, b] - zero, 'k-', lw=1)

# --- Draw vertical lines at every tick ---
for t in tick_x:
    ax_plot.axvline(t, color='gray', lw=0.5)

# --- Set tick labels ---
ax_plot.set_xticks(tick_x)
ax_plot.set_xticklabels(tick_labels)

# --- Set axis limits ---
ax_plot.set_xlim(x[0], x[-1])
ax_plot.set_ylim(-10, 10)
ax_plot.set_xlabel('k-point')
ax_plot.set_ylabel('Energy (eV)')
ax_plot.set_title('Band structure of fcc Al (PBEsol, aligned to Fermi energy)')

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=150)
print(f"Saved plot to {OUTPUT_PDF}")

# --- Print band edges and gap ---
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f"\nVBM = {vbm:.4f} eV (relative to Fermi)")
print(f"CBM = {cbm:.4f} eV (relative to Fermi)")
print(f"Band gap = {gap:.4f} eV")
print(f"Fermi energy = {fermi_eV:.4f} eV")
