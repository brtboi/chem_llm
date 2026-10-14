#!/usr/bin/env python3
"""
plot_bands.py -- Plot the band structure for calculations/000 (Cu, metal).

Uses:
  - chem_llm.kpath.plot_axis for x-axis, energies, ticks, segments (INVARIANT 4)
  - chem_llm.qe_output.read_pw_out for Fermi energy and n_occupied (INVARIANT 2)
  - chem_llm.qe_output.band_edges for VBM/CBM/gap

Cu is a metal (band_gap_eV=0.0 from Materials Project), so we align to the
Fermi energy from pw.out rather than the VBM.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# --- Load band data ---
ax = plot_axis('calculations/000/kpath.json', 'calculations/000/000.bands.dat')
x = ax['x']          # path length (arbitrary units)
E = ax['energies']   # shape (n_kpts, n_bands), in eV

# --- Read SCF output for Fermi energy and occupied-band count ---
pw = read_pw_out('calculations/000/pw.out')
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']

print(f"n_occupied = {n_occupied}")
print(f"Fermi energy = {fermi_eV:.4f} eV")

# --- Band edges (for reference) ---
edges = band_edges(E, n_occupied)
print(f"VBM = {edges['vbm']:.4f} eV")
print(f"CBM = {edges['cbm']:.4f} eV")
print(f"Band gap = {edges['gap']:.4f} eV")

# --- Align to Fermi energy (Cu is a metal) ---
zero = fermi_eV

# --- Plot ---
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Draw each band as one line per segment (never across a discontinuity)
for s in ax['segments']:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], E[s, b] - zero, 'k-', lw=1)

# Vertical lines at every tick
for t in ax['tick_x']:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Tick labels
plt_ax.set_xticks(ax['tick_x'])
plt_ax.set_xticklabels(ax['tick_labels'])

# Axis limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)

# Labels and title
plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title('Cu band structure (PBEsol, aligned to Fermi energy)')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf')
print("Saved calculations/000/bands_000.pdf")
