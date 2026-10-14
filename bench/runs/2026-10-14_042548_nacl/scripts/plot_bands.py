#!/usr/bin/env python3
"""
plot_bands.py - Plot the band structure for calculations/000.

Uses chem_llm.kpath.plot_axis for the x-axis, ticks, and line segments.
Aligns energies to the VBM (NaCl is an insulator).
Prints VBM, CBM, and band gap.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# --- Read the band data and k-path axis ---
kpath_json = 'calculations/000/kpath.json'
bands_dat = 'calculations/000/000.bands.dat'

ax = plot_axis(kpath_json, bands_dat)
x = ax['x']          # path length (arbitrary units)
E = ax['energies']   # shape (n_kpts, n_bands), in eV

# --- Read SCF output for electron count and Fermi energy ---
pw = read_pw_out('calculations/000/pw.out')
n_occupied = pw['n_occupied']
print(f"Number of occupied bands: {n_occupied}")

# Fermi energy may be None for insulators with fixed occupations
fermi = pw['fermi_energy_eV']
if fermi is not None:
    print(f"Fermi energy: {fermi:.4f} eV")
else:
    print("Fermi energy: not reported (insulator with fixed occupations)")

# --- Align to VBM (NaCl is an insulator) ---
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f"VBM: {vbm:.4f} eV")
print(f"CBM: {cbm:.4f} eV")
print(f"Band gap: {gap:.4f} eV")

# --- Plot ---
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in ax['segments']:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], E[s, b] - vbm, 'k-', lw=1)

# Draw vertical lines at every tick
for t in ax['tick_x']:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Set tick labels
plt_ax.set_xticks(ax['tick_x'])
plt_ax.set_xticklabels(ax['tick_labels'])

# Set axis limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)
plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV) relative to VBM')
plt_ax.set_title('NaCl Band Structure (PBEsol, scalar-relativistic)')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print("Saved calculations/000/bands_000.pdf")
