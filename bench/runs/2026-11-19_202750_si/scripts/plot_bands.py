#!/usr/bin/env python3
"""STEP 7: Plot the band structure for calculations/000.

Reads the bands.dat file produced by bands.x, aligns energies to the VBM
(semiconductor: occupations='fixed'), and saves a PDF plot.

INVARIANT 2: n_occupied comes from read_pw_out (pw.out), never assumed.
INVARIANT 4: x-axis, ticks, and line segments come from plot_axis(kpath.json, bands.dat).
"""
import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.dft.kpath import plot_axis
from chem_llm.dft.qe_output import read_pw_out, band_edges

CALC_DIR = 'calculations/000'
KPATH_JSON = f'{CALC_DIR}/kpath.json'
BANDS_DAT = f'{CALC_DIR}/000.bands.dat'
PW_OUT = f'{CALC_DIR}/pw.out'
OUT_PDF = f'{CALC_DIR}/bands_000.pdf'

# --- read SCF output for n_occupied and fermi energy ---
pw = read_pw_out(PW_OUT)
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']
print(f'n_occupied = {n_occupied}')
print(f'fermi_energy_eV = {fermi_eV}')

# --- get x-axis, energies, ticks, segments from plot_axis ---
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']          # path length array, shape (n_kpts,)
E = ax['energies']   # band energies in eV, shape (n_kpts, n_bands)
tick_x = ax['tick_x']
tick_labels = ax['tick_labels']
segments = ax['segments']

print(f'x shape: {x.shape}, E shape: {E.shape}')
print(f'n_ticks: {len(tick_x)}, n_segments: {len(segments)}')

# --- align to VBM (Si is a semiconductor, occupations='fixed') ---
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f'VBM = {vbm:.4f} eV')
print(f'CBM = {cbm:.4f} eV')
print(f'Band gap = {gap:.4f} eV')

# --- plot ---
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Draw each band as one line per segment (never across a discontinuity)
for s in segments:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], E[s, b] - vbm, 'k-', lw=1)

# Vertical lines at every tick
for t in tick_x:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Tick labels
plt_ax.set_xticks(tick_x)
plt_ax.set_xticklabels(tick_labels)

# Axis limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)
plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title('Si band structure (PBE, scalar-relativistic, no spin-orbit)\n'
                 f'VBM=0, gap={gap:.3f} eV')

plt.tight_layout()
plt.savefig(OUT_PDF, dpi=150)
print(f'Saved plot to {OUT_PDF}')
