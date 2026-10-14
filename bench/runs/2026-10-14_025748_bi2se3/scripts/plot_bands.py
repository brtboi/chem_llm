#!/usr/bin/env python3
"""
plot_bands.py -- Plot the band structure for calculations/000 (Bi2Se3, spin-orbit).

Reads the SCF output (pw.out) via chem_llm.qe_output to get n_occupied and
fermi_energy_eV, then uses chem_llm.kpath.plot_axis to build the x-axis,
energies, ticks and line segments from kpath.json and 000.bands.dat.

Bi2Se3 is a narrow-gap semiconductor (gap ~0.55 eV), so we align to the VBM.
With spin-orbit (noncolin), each band holds 1 electron, so n_occupied = n_electrons.
"""

import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# Paths
KPATH_JSON = 'calculations/000/kpath.json'
BANDS_DAT = 'calculations/000/000.bands.dat'
PW_OUT = 'calculations/000/pw.out'
OUTPUT_PDF = 'calculations/000/bands_000.pdf'

# Get the x-axis, energies, ticks and line segments from plot_axis
# This also checks that bands were computed on kpath.json's k-points
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']          # path length (array of shape (n_kpts,))
E = ax['energies']   # energies in eV (array of shape (n_kpts, n_bands))
segments = ax['segments']  # list of slice objects, one per continuous segment
tick_x = ax['tick_x']      # list of tick positions (floats)
tick_labels = ax['tick_labels']  # list of tick labels (strings)

# Read SCF output to get n_occupied and fermi_energy_eV
pw = read_pw_out(PW_OUT)
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']

print(f"n_occupied = {n_occupied}")
if fermi_eV is not None:
    print(f"Fermi energy = {fermi_eV:.4f} eV")
else:
    print("Fermi energy: not reported (occupations='fixed', no smearing)")

# Bi2Se3 is a narrow-gap semiconductor -> align to VBM
# band_edges returns vbm, cbm, gap in eV (relative to the raw energies in E)
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']

print(f"VBM = {vbm:.4f} eV")
print(f"CBM = {cbm:.4f} eV")
print(f"Band gap = {gap:.4f} eV")

# Align energies to VBM (set VBM = 0)
E_aligned = E - vbm

# Create the plot
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in segments:
    for b in range(E_aligned.shape[1]):
        plt_ax.plot(x[s], E_aligned[s, b], 'k-', lw=1)

# Draw a vertical line at every tick
for t in tick_x:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Label the ticks
plt_ax.set_xticks(tick_x)
plt_ax.set_xticklabels(tick_labels)

# Set x-limits to the full path
plt_ax.set_xlim(x[0], x[-1])

# Restrict y-axis to about +/-10 eV
plt_ax.set_ylim(-10, 10)

# Labels and title
plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title('Bi2Se3 band structure (spin-orbit, VBM = 0)')

plt.tight_layout()
plt.savefig(OUTPUT_PDF, bbox_inches='tight')
print(f"Saved plot to {OUTPUT_PDF}")
