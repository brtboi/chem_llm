#!/usr/bin/env python3
"""
STEP 7: Plot the band structure for calculations/000 (rutile TiO2, mp-2657).

Uses:
  - chem_llm.dft.kpath.plot_axis for x-axis, ticks, segments (INVARIANT 4)
  - chem_llm.dft.qe_output.read_pw_out for n_occupied, fermi_energy (INVARIANT 2)
  - chem_llm.dft.qe_output.band_edges for VBM/CBM/gap (INVARIANT 2)
  - matplotlib for the plot

Aligns to VBM (TiO2 is an insulator, gap ~1.78 eV from MP).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.dft.kpath import plot_axis
from chem_llm.dft.qe_output import read_pw_out, band_edges

# --- Load data ---
kpath_json = 'calculations/000/kpath.json'
bands_dat = 'calculations/000/000.bands.dat'
pw_out = 'calculations/000/pw.out'

# plot_axis checks that bands.dat was computed on kpath.json's k-points (INVARIANT 4)
ax = plot_axis(kpath_json, bands_dat)
x = ax['x']
E = ax['energies']  # shape (n_kpts, n_bands), in eV

# Read pw.out for n_occupied and fermi_energy (INVARIANT 2)
pw = read_pw_out(pw_out)
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']
print(f"n_occupied = {n_occupied}")
if fermi_eV is not None:
    print(f"fermi_energy = {fermi_eV:.4f} eV")
else:
    print("fermi_energy = None (insulator, fixed occupations)")

# Band edges (INVARIANT 2): VBM, CBM, gap
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f"VBM = {vbm:.4f} eV")
print(f"CBM = {cbm:.4f} eV")
print(f"Band gap = {gap:.4f} eV")

# Align to VBM (insulator)
zero = vbm

# --- Plot ---
fig, ax_plot = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in ax['segments']:
    for b in range(E.shape[1]):
        ax_plot.plot(x[s], E[s, b] - zero, 'k-', lw=1)

# Draw vertical line at every tick
for t in ax['tick_x']:
    ax_plot.axvline(t, color='gray', lw=0.5)

# Label ticks from plot_axis
ax_plot.set_xticks(ax['tick_x'])
ax_plot.set_xticklabels(ax['tick_labels'])

# Set x-limits to full path
ax_plot.set_xlim(x[0], x[-1])

# Restrict y-axis to about +/-10 eV
ax_plot.set_ylim(-10, 10)

# Labels and title
ax_plot.set_xlabel('k-point')
ax_plot.set_ylabel('Energy (eV)')
ax_plot.set_title('Rutile TiO2 (mp-2657) - PBE PAW band structure\n' +
                  f'VBM=0, gap={gap:.2f} eV, n_occ={n_occupied}')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print("Saved calculations/000/bands_000.pdf")
