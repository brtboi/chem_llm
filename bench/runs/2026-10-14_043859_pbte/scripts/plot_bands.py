#!/usr/bin/env python3
"""
plot_bands.py - Plot the band structure for calculations/000 (PbTe, spin-orbit).

Uses:
  - chem_llm.kpath.plot_axis for x-axis, energies, ticks, segments (INVARIANT 4)
  - chem_llm.qe_output.read_pw_out for n_occupied, fermi_energy_eV (INVARIANT 2)
  - chem_llm.qe_output.band_edges for VBM/CBM/gap
  - matplotlib for the plot

PbTe is a semiconductor (gap ~0.8 eV from Materials Project), so we align to the VBM.
With spin-orbit (noncolin), each band holds 1 electron, so n_occupied = 46 (from pw.out).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# Paths
KPATH_JSON = 'calculations/000/kpath.json'
BANDS_DAT = 'calculations/000/000.bands.dat'
PW_OUT = 'calculations/000/pw.out'
OUTPUT_PDF = 'calculations/000/bands_000.pdf'

# Get x-axis, energies, ticks, segments from plot_axis (INVARIANT 4)
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']
E = ax['energies']  # shape: (n_kpts, n_bands)
print(f"Energies shape: {E.shape}")
print(f"Number of k-points: {len(x)}")

# Read pw.out to get n_occupied and fermi_energy_eV (INVARIANT 2)
pw = read_pw_out(PW_OUT)
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']
print(f"n_occupied: {n_occupied}")
if fermi_eV is not None:
    print(f"Fermi energy: {fermi_eV:.4f} eV")
else:
    print("Fermi energy: not available (fixed occupations, no smearing)")

# PbTe is a semiconductor, so align to VBM
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f"VBM: {vbm:.4f} eV")
print(f"CBM: {cbm:.4f} eV")
print(f"Band gap: {gap:.4f} eV")

# Align energies to VBM
E_aligned = E - vbm

# Create the plot
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in ax['segments']:
    for b in range(E_aligned.shape[1]):
        plt_ax.plot(x[s], E_aligned[s, b], 'k-', lw=1)

# Draw vertical lines at every tick
for t in ax['tick_x']:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Label the ticks
plt_ax.set_xticks(ax['tick_x'])
plt_ax.set_xticklabels(ax['tick_labels'])

# Set axis limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)  # Restrict to +/-10 eV around VBM

# Labels and title
plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV) relative to VBM')
plt_ax.set_title('PbTe Band Structure (spin-orbit, majority spin)')

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=150, bbox_inches='tight')
print(f"\nSaved plot to {OUTPUT_PDF}")
print(f"\nSummary:")
print(f"  VBM: {vbm:.4f} eV")
print(f"  CBM: {cbm:.4f} eV")
print(f"  Band gap: {gap:.4f} eV")
