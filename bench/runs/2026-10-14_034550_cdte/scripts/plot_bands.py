#!/usr/bin/env python
"""
Plot the band structure for calculations/000 (unperturbed CdTe zincblende).

Uses chem_llm.kpath.plot_axis for the x-axis, ticks, and line segments,
and chem_llm.qe_output.read_pw_out for the number of occupied bands.
Aligns energies to the VBM (semiconductor, not a metal).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# Paths
kpath_json = 'calculations/000/kpath.json'
bands_dat = 'calculations/000/000.bands.dat'
pw_out = 'calculations/000/pw.out'
output_pdf = 'calculations/000/bands_000.pdf'

# Get x-axis, energies, ticks, and line segments from plot_axis
# This also verifies that bands.x computed the same k-points as kpath.json
ax = plot_axis(kpath_json, bands_dat)
x = ax['x']          # path length (A^-1 or similar)
E = ax['energies']   # shape (n_kpts, n_bands), in eV

# Read pw.out to get the number of occupied bands
pw = read_pw_out(pw_out)
n_occupied = pw['n_occupied']
print(f"Number of occupied bands: {n_occupied}")
print(f"Number of electrons: {pw['n_electrons']}")
print(f"Fermi energy: {pw['fermi_energy_eV']} eV")

# CdTe is a semiconductor (band gap ~0.59 eV from MP), so align to VBM
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']

print(f"VBM: {vbm:.4f} eV")
print(f"CBM: {cbm:.4f} eV")
print(f"Band gap: {gap:.4f} eV")

# Align energies to VBM (VBM at 0 eV)
E_aligned = E - vbm

# Plot
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in ax['segments']:
    for b in range(E_aligned.shape[1]):
        plt_ax.plot(x[s], E_aligned[s, b], 'k-', lw=1)

# Draw a vertical line at every tick
for t in ax['tick_x']:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Label the ticks from plot_axis's ticks
plt_ax.set_xticks(ax['tick_x'])
plt_ax.set_xticklabels(ax['tick_labels'])

# Set x-limits to the full path
plt_ax.set_xlim(x[0], x[-1])

# Restrict y-axis to about +/-10 eV
plt_ax.set_ylim(-10, 10)

# Labels and title
plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title(f'CdTe zincblende band structure (VBM at 0 eV)\nGap: {gap:.3f} eV')

plt.tight_layout()
plt.savefig(output_pdf, dpi=150)
print(f"Saved plot to {output_pdf}")
