#!/usr/bin/env python3
"""
Plot the band structure for calculations/000 (graphite C).

Uses chem_llm.kpath.plot_axis for the x-axis, ticks, and line segments,
and chem_llm.qe_output.read_pw_out for the Fermi energy and occupied-band count.
Graphite is a semimetal -> align to the Fermi energy.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# Paths
KPATH_JSON = 'calculations/000/kpath.json'
BANDS_DAT = 'calculations/000/000.bands.dat'
PW_OUT = 'calculations/000/pw.out'
OUTPUT_PDF = 'calculations/000/bands_000.pdf'

# Get x-axis, energies, ticks, and segments from plot_axis
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']
E = ax['energies']  # shape: (n_kpts, n_bands)

print(f"Number of k-points: {len(x)}")
print(f"Number of bands: {E.shape[1]}")

# Read pw.out for Fermi energy and occupied-band count
pw = read_pw_out(PW_OUT)
print(f"Number of electrons: {pw['n_electrons']}")
print(f"Number of occupied bands: {pw['n_occupied']}")
print(f"Fermi energy: {pw['fermi_energy_eV']} eV")

# Graphite is a semimetal -> align to Fermi energy
zero = pw['fermi_energy_eV']
print(f"Aligning to Fermi energy: {zero:.4f} eV")

# Also compute VBM/CBM for reference
edges = band_edges(E, pw['n_occupied'])
print(f"VBM: {edges['vbm']:.4f} eV")
print(f"CBM: {edges['cbm']:.4f} eV")
print(f"Band gap: {edges['gap']:.4f} eV")

# Plot
fig, ax_plot = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in ax['segments']:
    for b in range(E.shape[1]):
        ax_plot.plot(x[s], E[s, b] - zero, 'k-', lw=1)

# Draw vertical lines at every tick
for t in ax['tick_x']:
    ax_plot.axvline(t, color='gray', lw=0.5)

# Label ticks
ax_plot.set_xticks(ax['tick_x'])
ax_plot.set_xticklabels(ax['tick_labels'])

# Set axis limits
ax_plot.set_xlim(x[0], x[-1])
ax_plot.set_ylim(-10, 10)
ax_plot.set_xlabel('k-point')
ax_plot.set_ylabel('Energy (eV)')
ax_plot.set_title('Graphite C band structure (aligned to Fermi energy)')
ax_plot.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=150)
print(f"\nSaved plot to {OUTPUT_PDF}")
print(f"\nSummary:")
print(f"  Fermi energy: {zero:.4f} eV")
print(f"  VBM: {edges['vbm']:.4f} eV")
print(f"  CBM: {edges['cbm']:.4f} eV")
print(f"  Band gap: {edges['gap']:.4f} eV")
