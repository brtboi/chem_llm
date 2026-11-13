#!/usr/bin/env python
"""
Plot the band structure for calculations/000 (TiO2 rutile, P4_2/mnm).

Uses:
  - chem_llm.kpath.plot_axis for x-axis, energies, ticks, and line segments
  - chem_llm.qe_output.read_pw_out for n_occupied and fermi_energy_eV
  - chem_llm.qe_output.band_edges for VBM/CBM/gap

Aligns to VBM (semiconductor, gap ~1.77 eV from MP).
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

# Get x-axis, energies, ticks, and segments from plot_axis
# This also verifies that bands.dat was computed on the k-points in kpath.json
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']           # path length (arbitrary units)
E = ax['energies']    # shape (n_kpts, n_bands), in eV
segments = ax['segments']  # list of slice objects, one per continuous segment
tick_x = ax['tick_x']      # list of x positions for high-symmetry ticks
tick_labels = ax['tick_labels']  # list of labels (e.g. 'GAMMA', 'X|L', etc.)

print(f"k-points: {len(x)}, bands: {E.shape[1]}")
print(f"Segments: {len(segments)}")
print(f"Ticks: {list(zip(tick_x, tick_labels))}")

# Read SCF output to get n_occupied and fermi energy
pw = read_pw_out(PW_OUT)
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']

# For fixed occupations (insulator), QE does not print a Fermi energy line,
# so fermi_eV is None. That's fine -- we align to VBM instead.
if fermi_eV is not None:
    print(f"n_occupied = {n_occupied}, fermi_energy = {fermi_eV:.4f} eV")
else:
    print(f"n_occupied = {n_occupied} (no Fermi energy printed -- fixed occupations)")

# Align to VBM (semiconductor)
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f"VBM = {vbm:.4f} eV, CBM = {cbm:.4f} eV, gap = {gap:.4f} eV")

# Shift energies so VBM is at 0
eV_shifted = E - vbm

# Plot
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Draw each band as one line per segment (never across a discontinuity)
for s in segments:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], eV_shifted[s, b], 'k-', lw=1)

# Draw vertical lines at every tick
for t in tick_x:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Set x-axis ticks and labels
plt_ax.set_xticks(tick_x)
plt_ax.set_xticklabels(tick_labels)

# Set limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)  # about +/-10 eV around VBM

# Labels and title
plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title('TiO2 rutile (P4$_2$/mnm) band structure\n' +
                 f'PBEsol, ecutwfc=92 Ry, gap={gap:.2f} eV')

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=150)
print(f"Saved plot to {OUTPUT_PDF}")

# Print summary
print(f"\nSummary:")
print(f"  VBM: {vbm:.4f} eV")
print(f"  CBM: {cbm:.4f} eV")
print(f"  Band gap: {gap:.4f} eV")
if fermi_eV is not None:
    print(f"  Fermi energy: {fermi_eV:.4f} eV")
