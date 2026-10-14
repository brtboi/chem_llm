#!/usr/bin/env python3
"""
plot_bands.py -- Plot the band structure for calculations/000 (GaAs, zincblende).

Uses:
  - chem_llm.kpath.plot_axis for x-axis, ticks, segments (INVARIANT 4)
  - chem_llm.qe_output.read_pw_out for n_occupied and fermi_energy_eV (INVARIANT 2)
  - chem_llm.qe_output.band_edges for VBM/CBM/gap
  - matplotlib for the plot

GaAs is a semiconductor (band_gap_eV ~ 0.18 eV from MP), so we align to the VBM.
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
x = ax['x']           # path length (arbitrary units)
E = ax['energies']    # shape (n_kpts, n_bands), in eV
segments = ax['segments']  # list of slice objects, one per continuous segment
tick_x = ax['tick_x']
tick_labels = ax['tick_labels']

print(f"k-path: {len(x)} k-points, {E.shape[1]} bands")
print(f"Ticks: {list(zip(tick_x, tick_labels))}")

# --- Read SCF output for electron count and Fermi energy ---
pw = read_pw_out('calculations/000/pw.out')
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']

# For a non-smearing (occupations='fixed') run, QE does not print a Fermi energy
# line, so fermi_energy_eV is None. That is fine: we align to the VBM instead.
if fermi_eV is not None:
    print(f"n_occupied = {n_occupied}, fermi_energy = {fermi_eV:.4f} eV")
else:
    print(f"n_occupied = {n_occupied}, fermi_energy = None (no smearing, aligning to VBM)")

# --- Align to VBM (semiconductor) ---
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f"VBM = {vbm:.4f} eV, CBM = {cbm:.4f} eV, gap = {gap:.4f} eV")

# Shift energies so VBM is at 0
E_shifted = E - vbm

# --- Plot ---
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in segments:
    for b in range(E_shifted.shape[1]):
        plt_ax.plot(x[s], E_shifted[s, b], 'k-', lw=1)

# Draw vertical lines at every tick
for t in tick_x:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Set x-axis ticks and labels
plt_ax.set_xticks(tick_x)
plt_ax.set_xticklabels(tick_labels)

# Set axis limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)  # about +/-10 eV around VBM

# Labels and title
plt_ax.set_xlabel('k-path')
plt_ax.set_ylabel('Energy (eV) relative to VBM')
plt_ax.set_title('GaAs (zincblende) band structure -- PBEsol, scalar-relativistic')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print("Saved calculations/000/bands_000.pdf")

print(f"\nSummary:")
print(f"  VBM = {vbm:.4f} eV")
print(f"  CBM = {cbm:.4f} eV")
print(f"  Band gap = {gap:.4f} eV")
