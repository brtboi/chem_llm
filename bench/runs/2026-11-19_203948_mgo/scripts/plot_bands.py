#!/usr/bin/env python
"""Plot the band structure for calculations/000 (MgO, mp-1265).

Reads:
  - calculations/000/kpath.json (MP's k-path for the unperturbed cell)
  - calculations/000/000.bands.dat (bands.x output, energies in eV)
  - calculations/000/pw.out (SCF output, for n_occupied and Fermi energy)

Writes:
  - calculations/000/bands_000.pdf

Prints VBM, CBM, and band gap (eV).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.dft.kpath import plot_axis
from chem_llm.dft.qe_output import read_pw_out, band_edges

# --- Read the band data ---
# plot_axis checks that bands.dat was computed on the k-points in kpath.json
ax = plot_axis('calculations/000/kpath.json', 'calculations/000/000.bands.dat')
x = ax['x']          # path length (A^-1) for each k-point
E = ax['energies']   # shape (n_kpts, n_bands), energies in eV

# --- Read SCF output for occupied-band count and Fermi energy ---
pw = read_pw_out('calculations/000/pw.out')
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']

print(f"n_occupied = {n_occupied}")
if fermi_eV is not None:
    print(f"Fermi energy = {fermi_eV:.4f} eV")
else:
    print("Fermi energy: not reported (fixed-occupation insulator run)")

# --- Align to VBM (MgO is a clear insulator) ---
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

# Draw each band as one line per segment (never across a discontinuity)
for seg in ax['segments']:
    for b in range(E.shape[1]):
        plt_ax.plot(x[seg], eV_shifted[seg, b], 'k-', lw=1)

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
plt_ax.set_xlabel('k-point (A$^{-1}$)')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title(f'MgO band structure (PBE, PAW)\nGap = {gap:.3f} eV (VBM at 0)')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print("Saved calculations/000/bands_000.pdf")
