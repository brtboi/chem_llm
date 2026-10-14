#!/usr/bin/env python3
"""
plot_bands.py -- Plot the band structure for calculations/000 (rutile TiO2, primitive cell).

Reads:
  - calculations/000/kpath.json  (k-path from chem_llm.kpath)
  - calculations/000/000.bands.dat (band energies from bands.x)
  - calculations/000/pw.out      (SCF output for n_occupied / Fermi energy)

Writes:
  - calculations/000/bands_000.pdf

Prints VBM, CBM, and band gap (eV).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# --- Read the k-path axis and band energies ---
ax = plot_axis('calculations/000/kpath.json', 'calculations/000/000.bands.dat')
x = ax['x']          # path length (arbitrary units)
E = ax['energies']   # shape (n_kpts, n_bands), in eV

# --- Read SCF output for occupied-band count and Fermi energy ---
pw = read_pw_out('calculations/000/pw.out')
n_occ = pw['n_occupied']
print(f"n_occupied = {n_occ}")

# Fermi energy may be None for a non-smearing (fixed occupations) run;
# for a semiconductor we align to the VBM anyway, so just report it if present.
if pw['fermi_energy_eV'] is not None:
    print(f"Fermi energy = {pw['fermi_energy_eV']:.4f} eV")
else:
    print("Fermi energy = N/A (fixed occupations, no smearing)")

# --- Align to VBM (semiconductor, not a metal) ---
edges = band_edges(E, n_occ)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f"VBM = {vbm:.4f} eV")
print(f"CBM = {cbm:.4f} eV")
print(f"Band gap = {gap:.4f} eV")

# --- Plot ---
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in ax['segments']:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], E[s, b] - vbm, 'k-', lw=1)

# Draw vertical lines at every tick
for t in ax['tick_x']:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Label the ticks
plt_ax.set_xticks(ax['tick_x'])
plt_ax.set_xticklabels(ax['tick_labels'])

# Set axis limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)
plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV) relative to VBM')
plt_ax.set_title('Rutile TiO$_2$ band structure (PBEsol, 6-atom primitive cell)')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print("Saved calculations/000/bands_000.pdf")
