#!/usr/bin/env python
"""
plot_bands.py -- Plot the band structure for calculations/000 (SrTiO3, unperturbed).

Uses:
  - chem_llm.kpath.plot_axis for x-axis, energies, ticks, segments (INVARIANT 4)
  - chem_llm.qe_output.read_pw_out for n_occupied and fermi_energy_eV (INVARIANT 2)
  - chem_llm.qe_output.band_edges for VBM/CBM/gap
  - matplotlib for the plot

Aligns to VBM (semiconductor, gap ~1.77 eV from MP).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# --- Paths ---
KPATH_JSON = 'calculations/000/kpath.json'
BANDS_DAT = 'calculations/000/000.bands.dat'
PW_OUT = 'calculations/000/pw.out'
OUTPUT_PDF = 'calculations/000/bands_000.pdf'

# --- Get x-axis, energies, ticks, segments from plot_axis (INVARIANT 4) ---
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']           # path length (array)
E = ax['energies']    # energies in eV, shape (nkpts, nbands)
tick_x = ax['tick_x']
tick_labels = ax['tick_labels']
segments = ax['segments']  # list of slice objects

print(f"k-points: {len(x)}, bands: {E.shape[1]}")
print(f"Ticks: {list(zip(tick_x, tick_labels))}")

# --- Read pw.out for n_occupied and fermi energy (INVARIANT 2) ---
pw = read_pw_out(PW_OUT)
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']
# fermi_eV is None for fixed-occupation runs (no smearing) -- guard the print
if fermi_eV is not None:
    print(f"n_occupied = {n_occupied}, fermi_energy = {fermi_eV:.4f} eV")
else:
    print(f"n_occupied = {n_occupied}, fermi_energy = N/A (fixed occupations)")

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

# Set limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)  # restrict to +/-10 eV around VBM

# Labels and title
plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV) relative to VBM')
plt_ax.set_title('SrTiO3 band structure (PBEsol, scalar-relativistic)\n' +
                 f'Gap = {gap:.3f} eV (VBM at 0 eV)')

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=150)
print(f"Saved plot to {OUTPUT_PDF}")

# --- Print summary ---
print(f"\n=== Band structure summary for calculations/000 ===")
print(f"VBM: {vbm:.4f} eV")
print(f"CBM: {cbm:.4f} eV")
print(f"Band gap: {gap:.4f} eV")
if fermi_eV is not None:
    print(f"Fermi energy (from pw.out): {fermi_eV:.4f} eV")
else:
    print("Fermi energy: N/A (fixed occupations, semiconductor)")
print(f"n_occupied: {n_occupied}")
