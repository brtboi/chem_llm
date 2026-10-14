#!/usr/bin/env python3
"""
plot_bands.py -- Plot the band structure for calculations/000 (hBN, mp-629015).

Uses:
  - chem_llm.kpath.plot_axis for x-axis, energies, ticks, and line segments
  - chem_llm.qe_output.read_pw_out for n_occupied and fermi_energy_eV
  - chem_llm.qe_output.band_edges for VBM/CBM/gap
  - matplotlib for the plot

hBN is an insulator (gap ~4 eV), so energies are aligned to the VBM.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# --- Load the band data and k-path axis ---
kpath_json = 'calculations/000/kpath.json'
bands_dat = 'calculations/000/000.bands.dat'

ax = plot_axis(kpath_json, bands_dat)
x = ax['x']          # path length (arbitrary units)
E = ax['energies']   # shape (n_kpts, n_bands), in eV

# --- Read SCF output for occupied-band count and Fermi energy ---
pw = read_pw_out('calculations/000/pw.out')
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']

print(f"n_occupied = {n_occupied}")
if fermi_eV is not None:
    print(f"Fermi energy = {fermi_eV:.4f} eV")
else:
    print("Fermi energy = None (insulator with fixed occupations; not reported by QE)")

# --- Align to VBM (insulator) ---
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']

print(f"VBM = {vbm:.4f} eV")
print(f"CBM = {cbm:.4f} eV")
print(f"Band gap = {gap:.4f} eV")

# Shift energies so VBM is at 0
eV_shifted = E - vbm

# --- Plot ---
fig, ax_plot = plt.subplots(figsize=(10, 6))

# Draw each band as one line per segment (no line across discontinuities)
for s in ax['segments']:
    for b in range(E.shape[1]):
        ax_plot.plot(x[s], eV_shifted[s, b], 'k-', lw=1)

# Vertical lines at each tick
for t in ax['tick_x']:
    ax_plot.axvline(t, color='gray', lw=0.5)

# Tick labels
ax_plot.set_xticks(ax['tick_x'])
ax_plot.set_xticklabels(ax['tick_labels'])

# Axis limits
ax_plot.set_xlim(x[0], x[-1])
ax_plot.set_ylim(-10, 10)
ax_plot.set_xlabel('k-point')
ax_plot.set_ylabel('Energy (eV) relative to VBM')
ax_plot.set_title('hBN band structure (PBEsol, scalar-relativistic)')

plt.tight_layout()
out_path = 'calculations/000/bands_000.pdf'
plt.savefig(out_path, dpi=150)
print(f"Saved plot to {out_path}")
