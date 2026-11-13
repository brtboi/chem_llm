"""
plot_bands.py -- Plot the band structure for calculations/000 (rutile TiO2, PBEsol).

Reads the bands.dat file produced by bands.x, aligns energies to the VBM
(semiconductor, nspin=1), and saves a PDF plot.

Uses:
  - chem_llm.dft.kpath.plot_axis for x-axis, ticks, segments (INVARIANT 4)
  - chem_llm.dft.qe_output.read_pw_out for n_occupied, fermi_energy_eV (INVARIANT 2)
  - chem_llm.dft.qe_output.band_edges for VBM/CBM/gap (INVARIANT 2)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.dft.kpath import plot_axis
from chem_llm.dft.qe_output import read_pw_out, band_edges

# --- Read the band data ---
kpath_json = 'calculations/000/kpath.json'
bands_dat = 'calculations/000/000.bands.dat'
pw_out = 'calculations/000/pw.out'

# plot_axis checks that bands.dat was computed on kpath.json's k-points
ax = plot_axis(kpath_json, bands_dat)
x = ax['x']          # path-length x-axis
E = ax['energies']   # shape (n_kpts, n_bands), in eV
segments = ax['segments']
tick_x = ax['tick_x']
tick_labels = ax['tick_labels']

print(f'Band data: {E.shape[0]} k-points, {E.shape[1]} bands')

# --- Read SCF output for electron count and Fermi energy ---
pw = read_pw_out(pw_out)
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']
# For a non-smearing SCF run (occupations='fixed'), QE does not print a Fermi energy line,
# so fermi_eV may be None. Guard the print accordingly.
if fermi_eV is not None:
    print(f'n_occupied = {n_occupied}, fermi_energy = {fermi_eV:.4f} eV')
else:
    print(f'n_occupied = {n_occupied}, fermi_energy = None (no smearing, not printed by QE)')

# --- Align to VBM (semiconductor, not a metal) ---
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f'VBM = {vbm:.4f} eV, CBM = {cbm:.4f} eV, gap = {gap:.4f} eV')

# Shift energies so VBM is at 0
eV_shifted = E - vbm

# --- Plot ---
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Draw each band as one line per segment (no line across a discontinuity)
for s in segments:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], eV_shifted[s, b], 'k-', lw=0.8)

# Vertical lines at every tick
for t in tick_x:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Tick labels
plt_ax.set_xticks(tick_x)
plt_ax.set_xticklabels(tick_labels)

# Axis limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)  # +/-10 eV around VBM

plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title('Rutile TiO$_2$ band structure (PBEsol, scalar-relativistic)')

plt.tight_layout()
out_pdf = 'calculations/000/bands_000.pdf'
plt.savefig(out_pdf, dpi=150)
print(f'Saved plot to {out_pdf}')

print(f'\nSummary:')
print(f'  VBM = {vbm:.4f} eV')
print(f'  CBM = {cbm:.4f} eV')
print(f'  Band gap = {gap:.4f} eV')
