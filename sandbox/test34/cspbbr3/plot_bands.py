#!/usr/bin/env python3
"""
plot_bands.py -- Plot the band structure for calculations/000 (CsPbBr3, spin-orbit).

Uses:
  - chem_llm.kpath.plot_axis for x-axis, energies, ticks, and line segments
  - chem_llm.qe_output.read_pw_out for n_occupied and fermi_energy_eV
  - chem_llm.qe_output.band_edges for VBM/CBM/gap
  - matplotlib for the plot

Aligns to the VBM (semiconductor, gap ~1.78 eV from MP).
Saves calculations/000/bands_000.pdf.
"""

import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# --- Paths ---
KPATH_JSON = 'calculations/000/kpath.json'
BANDS_DAT = 'calculations/000/000.bands.dat'
PW_OUT = 'calculations/000/pw.out'
OUTPUT_PDF = 'calculations/000/bands_000.pdf'

# --- Read band data (INVARIANT 4: plot_axis checks k-point consistency) ---
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']          # path length (array)
E = ax['energies']   # (n_kpoints, n_bands) in eV
segments = ax['segments']   # list of slice objects, one per continuous segment
tick_x = ax['tick_x']       # list of float positions
tick_labels = ax['tick_labels']  # list of strings

print(f"Band data: {E.shape[0]} k-points, {E.shape[1]} bands")
print(f"Path length: {x[0]:.3f} to {x[-1]:.3f}")
print(f"Ticks: {list(zip(tick_x, tick_labels))}")

# --- Read SCF output (INVARIANT 2: use chem_llm.qe_output, not hand-written regex) ---
pw = read_pw_out(PW_OUT)
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']  # None for fixed-occupations runs (no smearing)
print(f"\nFrom pw.out:")
print(f"  n_electrons = {pw['n_electrons']}")
print(f"  n_occupied  = {n_occupied}")
if fermi_eV is not None:
    print(f"  fermi_energy = {fermi_eV:.4f} eV")
else:
    print(f"  fermi_energy = None (fixed occupations, no smearing)")
print(f"  spin_orbit  = {pw['spin_orbit']}")

# --- Align to VBM (semiconductor) ---
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f"\nBand edges (before alignment):")
print(f"  VBM = {vbm:.4f} eV")
print(f"  CBM = {cbm:.4f} eV")
print(f"  Gap = {gap:.4f} eV")

# Shift so VBM is at 0 eV
E_shifted = E - vbm

# --- Plot ---
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Draw each band as one line per segment (never across a discontinuity)
for s in segments:
    for b in range(E_shifted.shape[1]):
        plt_ax.plot(x[s], E_shifted[s, b], 'k-', lw=1)

# Draw vertical lines at every tick
for t in tick_x:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Label the ticks
plt_ax.set_xticks(tick_x)
plt_ax.set_xticklabels(tick_labels)

# Set axis limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)  # +/-10 eV around VBM

# Labels and title
plt_ax.set_xlabel('k-path')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title('CsPbBr3 band structure (PBEsol, spin-orbit)\nVBM at 0 eV')

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=150)
print(f"\nSaved plot to {OUTPUT_PDF}")

# --- Print summary ---
print(f"\n=== Summary ===")
print(f"VBM = {vbm:.4f} eV (set to 0)")
print(f"CBM = {cbm:.4f} eV")
print(f"Band gap = {gap:.4f} eV")
if fermi_eV is not None:
    print(f"Fermi energy (from pw.out) = {fermi_eV:.4f} eV")
else:
    print(f"Fermi energy: not available (fixed occupations)")
