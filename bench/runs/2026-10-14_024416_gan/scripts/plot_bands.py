#!/usr/bin/env python3
"""
plot_bands.py -- Plot the band structure for calculations/000 (wurtzite GaN).

Uses chem_llm.kpath.plot_axis for the x-axis, ticks, and line segments (INVARIANT 4).
Uses chem_llm.qe_output.read_pw_out and band_edges for energy alignment (INVARIANT 2).
Aligns to the VBM (GaN is a semiconductor, not a metal).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# --- Read the band energies and x-axis from plot_axis ---
# plot_axis checks that bands.x computed the same k-points as kpath.json
ax = plot_axis('calculations/000/kpath.json', 'calculations/000/000.bands.dat')
x = ax['x']          # path length (arbitrary units)
E = ax['energies']   # shape (n_kpts, n_bands), in eV

# --- Read SCF output for n_occupied and Fermi energy ---
pw = read_pw_out('calculations/000/pw.out')
n_occupied = pw['n_occupied']
print(f"n_occupied = {n_occupied}")

# Fermi energy may be None for a semiconductor with fixed occupations
fermi = pw['fermi_energy_eV']
if fermi is not None:
    print(f"Fermi energy = {fermi:.4f} eV")
else:
    print("Fermi energy: not reported (semiconductor, fixed occupations)")

# --- Align to VBM (GaN is a semiconductor) ---
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

# Plot each band as one line per segment (never across a discontinuity)
for s in ax['segments']:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], eV_shifted[s, b], 'k-', lw=1)

# Draw vertical lines at every tick
for t in ax['tick_x']:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Set tick labels from plot_axis
plt_ax.set_xticks(ax['tick_x'])
plt_ax.set_xticklabels(ax['tick_labels'])

# Set axis limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)  # +/-10 eV around VBM

plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title('Wurtzite GaN band structure (PBEsol, scalar relativistic)')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print("Saved calculations/000/bands_000.pdf")
