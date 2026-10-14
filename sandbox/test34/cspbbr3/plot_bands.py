"""
plot_bands.py -- Plot the band structure for calculations/000 (CsPbBr3, spin-orbit).

Uses:
  - chem_llm.kpath.plot_axis for the x-axis, ticks, and line segments (INVARIANT 4)
  - chem_llm.qe_output.read_pw_out for n_occupied and Fermi energy (INVARIANT 2)
  - chem_llm.qe_output.band_edges for VBM/CBM/gap
  - matplotlib for the plot

The material is an insulator (gap ~1.6-2 eV without SO, ~0.4-1.6 eV with SO),
so we align to the VBM.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# --- Read the band data and k-path axis ---
kpath_json = 'calculations/000/kpath.json'
bands_dat = 'calculations/000/000.bands.dat'

ax = plot_axis(kpath_json, bands_dat)
x = ax['x']           # path length (in units of 2*pi/a or similar)
E = ax['energies']    # shape (n_kpts, n_bands), in eV

print(f"x range: {x[0]:.4f} to {x[-1]:.4f}")
print(f"E shape: {E.shape}")
print(f"Ticks: {ax['tick_labels']}")
print(f"Segments: {len(ax['segments'])}")

# --- Read pw.out for n_occupied and Fermi energy ---
pw = read_pw_out('calculations/000/pw.out')
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']
print(f"n_occupied = {n_occupied}")
if fermi_eV is not None:
    print(f"Fermi energy = {fermi_eV:.4f} eV")
else:
    print("Fermi energy = None (insulator with fixed occupations, no smearing)")

# --- Compute band edges (VBM, CBM, gap) ---
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f"VBM = {vbm:.4f} eV")
print(f"CBM = {cbm:.4f} eV")
print(f"Band gap = {gap:.4f} eV")

# --- Align to VBM (insulator) ---
zero = vbm

# --- Plot ---
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in ax['segments']:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], E[s, b] - zero, 'k-', lw=1)

# Draw vertical lines at every tick
for t in ax['tick_x']:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Set x-axis ticks and labels
plt_ax.set_xticks(ax['tick_x'])
plt_ax.set_xticklabels(ax['tick_labels'])

# Set axis limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)

# Labels and title
plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV) relative to VBM')
plt_ax.set_title('CsPbBr3 band structure (spin-orbit, PBEsol)')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print("Saved plot to calculations/000/bands_000.pdf")
