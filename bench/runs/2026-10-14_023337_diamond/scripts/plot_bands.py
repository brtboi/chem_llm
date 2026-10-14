#!/usr/bin/env python
"""
plot_bands.py -- Plot the band structure for calculations/000 (diamond C).

Uses chem_llm.kpath.plot_axis for the x-axis, ticks, and line segments
(INVARIANT 4), and chem_llm.qe_output.read_pw_out for the electron count
(INVARIANT 2).  Aligns energies to the VBM (diamond is a clear insulator).

Outputs: calculations/000/bands_000.pdf
Prints:  VBM, CBM, and band gap in eV.
"""

import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# --- Paths ---
CALC_DIR = 'calculations/000'
KPATH_JSON = f'{CALC_DIR}/kpath.json'
BANDS_DAT = f'{CALC_DIR}/000.bands.dat'
PW_OUT = f'{CALC_DIR}/pw.out'
OUTPUT_PDF = f'{CALC_DIR}/bands_000.pdf'

# --- Get x-axis, energies, ticks, segments from plot_axis (INVARIANT 4) ---
# plot_axis checks that bands.dat was computed on the same k-points as kpath.json
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']           # path length (A^-1 or similar), shape (N_kpts,)
E = ax['energies']    # energies in eV, shape (N_kpts, N_bands)
tick_x = ax['tick_x']          # list of x positions for high-symmetry ticks
tick_labels = ax['tick_labels']  # list of tick labels (e.g. 'GAMMA', 'X', 'U|K', ...)
segments = ax['segments']    # list of slice objects, one per continuous segment

print(f'k-path: {len(x)} k-points, {E.shape[1]} bands')
print(f'Ticks: {list(zip(tick_x, tick_labels))}')

# --- Read pw.out for electron count (INVARIANT 2) ---
pw = read_pw_out(PW_OUT)
n_occupied = pw['n_occupied']  # number of occupied bands (spin-unpolarized: n_electrons/2)
n_electrons = pw['n_electrons']
print(f'pw.out: n_electrons = {n_electrons}, n_occupied = {n_occupied}')

# INVARIANT 2: verify nbnd was large enough
# (nbnd=16 was set in setup_jobs.py; n_occupied should be 4 for 8 electrons)
assert n_occupied == 4, f'Expected 4 occupied bands, got {n_occupied}'

# --- Align to VBM (diamond is an insulator, not a metal) ---
edges = band_edges(E, n_occupied)
vbm = edges['vbm']  # eV
cbm = edges['cbm']  # eV
gap = edges['gap']  # eV
print(f'VBM = {vbm:.4f} eV')
print(f'CBM = {cbm:.4f} eV')
print(f'Band gap = {gap:.4f} eV')

# --- Plot ---
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in segments:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], E[s, b] - vbm, 'k-', lw=1)

# Draw vertical lines at every tick
for t in tick_x:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Label the ticks
plt_ax.set_xticks(tick_x)
plt_ax.set_xticklabels(tick_labels)

# Set x-limits to the full path
plt_ax.set_xlim(x[0], x[-1])

# Restrict y-axis to about +/-10 eV around VBM
plt_ax.set_ylim(-10, 10)

plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title(f'Diamond C band structure (PBEsol, VBM = 0 eV, gap = {gap:.2f} eV)')
plt_ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=150)
print(f'Saved plot to {OUTPUT_PDF}')
