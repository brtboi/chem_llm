#!/usr/bin/env python
"""STEP 7: Plot the band structure for calculations/000.

Reads the bands.dat file produced by bands.x, aligns energies to the VBM
(semiconductor, so VBM alignment is appropriate), and saves a PDF plot.

Uses:
  - chem_llm.dft.kpath.plot_axis for x-axis, ticks, segments (INVARIANT 4)
  - chem_llm.dft.qe_output.read_pw_out for n_occupied (INVARIANT 2)
  - chem_llm.dft.qe_output.band_edges for VBM/CBM/gap (INVARIANT 2)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.dft.kpath import plot_axis
from chem_llm.dft.qe_output import read_pw_out, band_edges

# --- Load the band data ---
kpath_json = 'calculations/000/kpath.json'
bands_dat = 'calculations/000/000.bands.dat'
pw_out = 'calculations/000/pw.out'

# plot_axis checks that bands.dat was computed on kpath.json's k-points (INVARIANT 4)
ax = plot_axis(kpath_json, bands_dat)
x = ax['x']          # path length (A^-1)
E = ax['energies']   # shape (n_kpts, n_bands), in eV

# --- Read SCF output for electron count (INVARIANT 2) ---
pw = read_pw_out(pw_out)
n_occupied = pw['n_occupied']
print(f"Number of electrons: {pw['n_electrons']}")
print(f"Number of occupied bands: {n_occupied}")

# Fermi energy may be None for a fixed-occupation run (no smearing)
if pw['fermi_energy_eV'] is not None:
    print(f"Fermi energy: {pw['fermi_energy_eV']:.4f} eV")
else:
    print("Fermi energy: not reported (fixed occupations, no smearing)")

# --- Align to VBM (Si is a semiconductor, not a metal) ---
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f"VBM: {vbm:.4f} eV")
print(f"CBM: {cbm:.4f} eV")
print(f"Band gap: {gap:.4f} eV")

# Shift energies so VBM is at 0
eV_shift = E - vbm

# --- Plot ---
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Draw each band as one line per segment (never across a discontinuity)
for s in ax['segments']:
    for b in range(eV_shift.shape[1]):
        plt_ax.plot(x[s], eV_shift[s, b], 'k-', lw=1)

# Draw vertical lines at every tick
for t in ax['tick_x']:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Set tick labels from plot_axis (they already read 'A|L' at discontinuities)
plt_ax.set_xticks(ax['tick_x'])
plt_ax.set_xticklabels(ax['tick_labels'])

# Set axis limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)  # +/-10 eV around VBM

# Labels and title
plt_ax.set_xlabel('k-path (A$^{-1}$)')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title('Si band structure (PBE, PAW, no spin-orbit)\n' +
                 f'Gap: {gap:.3f} eV (VBM at 0 eV)')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', dpi=150)
print("Saved calculations/000/bands_000.pdf")
