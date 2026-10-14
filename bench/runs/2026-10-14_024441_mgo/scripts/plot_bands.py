#!/usr/bin/env python3
"""
plot_bands.py -- Plot the band structure for calculations/000.

Uses chem_llm.kpath.plot_axis for the x-axis, ticks, and line segments,
and chem_llm.qe_output.read_pw_out for the Fermi energy / occupied-band count.
Aligns to the VBM (MgO is an insulator).
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

# Get x-axis, energies, ticks, and segments from plot_axis (INVARIANT 4)
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']
E = ax['energies']  # shape: (n_kpoints, n_bands)

# Read pw.out for electron count and Fermi energy (INVARIANT 2)
pw = read_pw_out(PW_OUT)
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']

print(f"Number of electrons: {pw['n_electrons']}")
print(f"Number of occupied bands: {n_occupied}")
if fermi_eV is not None:
    print(f"Fermi energy: {fermi_eV:.4f} eV")
else:
    print("Fermi energy: not reported (occupations='fixed', insulator)")

# Align to VBM (MgO is an insulator, not a metal)
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']

print(f"VBM: {vbm:.4f} eV")
print(f"CBM: {cbm:.4f} eV")
print(f"Band gap: {gap:.4f} eV")

# Plot
fig, ax_plot = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in ax['segments']:
    for b in range(E.shape[1]):
        ax_plot.plot(x[s], E[s, b] - vbm, 'k-', lw=1)

# Draw vertical lines at every tick
for t in ax['tick_x']:
    ax_plot.axvline(t, color='gray', lw=0.5)

# Label the ticks
ax_plot.set_xticks(ax['tick_x'])
ax_plot.set_xticklabels(ax['tick_labels'])

# Set axis limits
ax_plot.set_xlim(x[0], x[-1])
ax_plot.set_ylim(-10, 10)
ax_plot.set_xlabel('k-point')
ax_plot.set_ylabel('Energy (eV)')
ax_plot.set_title('MgO Band Structure (PBEsol, aligned to VBM)')

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=150)
print(f"Saved plot to {OUTPUT_PDF}")
print("Done.")
