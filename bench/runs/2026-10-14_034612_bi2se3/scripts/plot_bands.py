#!/usr/bin/env python
"""
plot_bands.py -- Plot the band structure for calculations/000 (Bi2Se3, spin-orbit).

Uses chem_llm.kpath.plot_axis for the x-axis, ticks, and line segments (INVARIANT 4),
and chem_llm.qe_output.read_pw_out for the electron count and Fermi energy (INVARIANT 2).

Bi2Se3 is a narrow-gap semiconductor (gap ~0.35 eV experimental, ~0.55 eV from MP).
With spin-orbit, the gap is well-resolved. We align to the VBM (not the Fermi energy)
since this is a semiconductor, not a metal.

Each band holds ONE electron (spin-orbit), so n_occupied = n_electrons = 118.
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
x = ax['x']          # path length (in some unit, consistent across the plot)
E = ax['energies']   # shape (n_kpts, n_bands), in eV
segments = ax['segments']  # list of slice objects, one per continuous segment
tick_x = ax['tick_x']      # list of tick positions
tick_labels = ax['tick_labels']  # list of tick labels (e.g. 'GAMMA', 'T', 'H_2|H_0', ...)

print(f"x range: {x[0]:.4f} to {x[-1]:.4f}")
print(f"E shape: {E.shape}")
print(f"n segments: {len(segments)}")
print(f"ticks: {list(zip(tick_x, tick_labels))}")

# --- Read pw.out to get n_occupied and Fermi energy (INVARIANT 2) ---
pw = read_pw_out(PW_OUT)
n_electrons = pw['n_electrons']
n_occupied = pw['n_occupied']  # already 1 electron/band with spin-orbit
fermi_eV = pw['fermi_energy_eV']
spin_orbit = pw['spin_orbit']

print(f"n_electrons = {n_electrons}")
print(f"n_occupied = {n_occupied}")
print(f"fermi_energy_eV = {fermi_eV}")
print(f"spin_orbit = {spin_orbit}")

# --- Align to VBM (semiconductor, not a metal) ---
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']

print(f"VBM = {vbm:.4f} eV")
print(f"CBM = {cbm:.4f} eV")
print(f"Band gap = {gap:.4f} eV")

# --- Plot ---
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in segments:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], E[s, b] - vbm, 'k-', lw=1)

# Draw a vertical line at every tick
for t in tick_x:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Set tick labels
plt_ax.set_xticks(tick_x)
plt_ax.set_xticklabels(tick_labels)

# Set x-limits to the full path
plt_ax.set_xlim(x[0], x[-1])

# Restrict y-axis to about +/-10 eV
plt_ax.set_ylim(-10, 10)

# Labels and title
plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV) relative to VBM')
plt_ax.set_title('Bi2Se3 band structure (spin-orbit, PBESOL)\nAligned to VBM')

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=150)
print(f"\nSaved plot to {OUTPUT_PDF}")

# --- Print summary ---
print(f"\n=== Summary ===")
print(f"VBM: {vbm:.4f} eV")
print(f"CBM: {cbm:.4f} eV")
print(f"Band gap: {gap:.4f} eV")
if fermi_eV is not None:
    print(f"Fermi energy: {fermi_eV:.4f} eV")
else:
    print("Fermi energy: not available (no smearing used)")
