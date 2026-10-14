#!/usr/bin/env python3
"""
plot_bands.py -- Plot the band structure for calculations/000 (ZnO, wurtzite).

Uses:
  - chem_llm.kpath.plot_axis for x-axis, ticks, segments (INVARIANT 4)
  - chem_llm.qe_output.read_pw_out for n_occupied, fermi_energy (INVARIANT 2)
  - chem_llm.qe_output.band_edges for VBM/CBM/gap
  - matplotlib for the plot

Aligns to VBM (ZnO is a semiconductor, not a metal).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# --- Load band data and k-path axis ---
kpath_json = 'calculations/000/kpath.json'
bands_dat = 'calculations/000/000.bands.dat'

ax = plot_axis(kpath_json, bands_dat)
x = ax['x']          # path length (A^-1 or similar)
E = ax['energies']   # shape (nkpts, nbands), in eV

print(f"Band data: {E.shape[0]} k-points, {E.shape[1]} bands")
print(f"Path length: {x[0]:.4f} to {x[-1]:.4f}")

# --- Read SCF output for electron count and Fermi energy ---
pw = read_pw_out('calculations/000/pw.out')
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']

# fermi_eV is None for non-metallic runs with occupations='fixed'
# (QE does not print a Fermi energy line in that case)
if fermi_eV is not None:
    print(f"n_occupied = {n_occupied}, fermi_energy = {fermi_eV:.4f} eV")
else:
    print(f"n_occupied = {n_occupied}, fermi_energy = None (non-metallic, fixed occupations)")

# INVARIANT 2: check nbnd >= n_occupied
nbnd_in_file = E.shape[1]
assert nbnd_in_file >= n_occupied, f"nbnd={nbnd_in_file} < n_occupied={n_occupied}"
print(f"nbnd={nbnd_in_file} >= n_occupied={n_occupied} -- OK")

# --- Align to VBM (ZnO is a semiconductor) ---
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f"VBM = {vbm:.4f} eV, CBM = {cbm:.4f} eV, gap = {gap:.4f} eV")

# Shift energies so VBM is at 0
eV_shifted = E - vbm

# --- Plot ---
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Draw each band as one line per segment (never across a discontinuity)
for s in ax['segments']:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], eV_shifted[s, b], 'k-', lw=1)

# Draw vertical lines at every tick
for t in ax['tick_x']:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Set x-ticks and labels from plot_axis
plt_ax.set_xticks(ax['tick_x'])
plt_ax.set_xticklabels(ax['tick_labels'])

# Set axis limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)  # +/-10 eV around VBM

# Labels and title
plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title('ZnO (wurtzite) band structure -- PBEsol, VBM-aligned')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print("Saved calculations/000/bands_000.pdf")

print(f"\nSummary:")
print(f"  VBM = {vbm:.4f} eV")
print(f"  CBM = {cbm:.4f} eV")
print(f"  Band gap = {gap:.4f} eV")
if fermi_eV is not None:
    print(f"  Fermi energy = {fermi_eV:.4f} eV")
else:
    print(f"  Fermi energy = N/A (non-metallic, fixed occupations)")
