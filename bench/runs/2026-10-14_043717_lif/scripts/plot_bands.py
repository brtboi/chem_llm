#!/usr/bin/env python3
"""
plot_bands.py -- Plot the band structure for calculations/000.

Uses:
  - chem_llm.kpath.plot_axis for x-axis, energies, ticks, segments
  - chem_llm.qe_output.read_pw_out for n_occupied and Fermi energy
  - chem_llm.qe_output.band_edges for VBM/CBM/gap
  - matplotlib for the plot

LiF is an insulator (gap ~8.7 eV), so we align to the VBM.
fermi_energy_eV is None for fixed-occupation runs (no smearing), so we
handle that case explicitly.
"""

import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# --- Load the band data and axis info ---
kpath_json = 'calculations/000/kpath.json'
bands_dat = 'calculations/000/000.bands.dat'

ax = plot_axis(kpath_json, bands_dat)
x = ax['x']          # path length array
E = ax['energies']   # shape (nkpts, nbands), in eV

# --- Read SCF output for electron count and Fermi energy ---
pw = read_pw_out('calculations/000/pw.out')
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']

print(f"n_occupied = {n_occupied}")
if fermi_eV is not None:
    print(f"Fermi energy = {fermi_eV:.4f} eV")
else:
    print("Fermi energy = None (fixed occupations, no smearing -- expected for insulator)")

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
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (no line across discontinuities)
for s in ax['segments']:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], eV_shifted[s, b], 'k-', lw=1)

# Draw vertical lines at every tick
for t in ax['tick_x']:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Label the ticks
plt_ax.set_xticks(ax['tick_x'])
plt_ax.set_xticklabels(ax['tick_labels'])

# Set axis limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)  # +/-10 eV around VBM

# Labels and title
plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title('LiF band structure (PBEsol, VBM-aligned)')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print("Saved calculations/000/bands_000.pdf")
