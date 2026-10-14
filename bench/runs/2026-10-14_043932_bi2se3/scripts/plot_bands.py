"""
plot_bands.py -- Plot the band structure for calculations/000 (Bi2Se3, spin-orbit).

Uses:
  - chem_llm.kpath.plot_axis for x-axis, ticks, segments (INVARIANT 4)
  - chem_llm.qe_output.read_pw_out for n_occupied, fermi_energy_eV (INVARIANT 2)
  - chem_llm.qe_output.band_edges for VBM/CBM/gap

Bi2Se3 is a narrow-gap semiconductor (band_gap_eV=0.553 from MP).
With spin-orbit, each band holds 1 electron, so n_occupied = n_electrons.
Align to VBM (semiconductor, not a metal).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# Read the band data and k-path axis
ax = plot_axis('calculations/000/kpath.json', 'calculations/000/000.bands.dat')
x = ax['x']          # path length (A^-1 or similar)
E = ax['energies']   # shape (n_kpts, n_bands), eV

# Read SCF output for electron count and Fermi energy
pw = read_pw_out('calculations/000/pw.out')
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']

print(f"n_occupied = {n_occupied}")
print(f"Fermi energy = {fermi_eV:.4f} eV")
print(f"Number of bands in data: {E.shape[1]}")

# INVARIANT 2: with spin-orbit, n_occupied = n_electrons (1 e/band)
# Check that we have enough bands
assert E.shape[1] >= n_occupied, f"Not enough bands: {E.shape[1]} < {n_occupied}"

# Band edges (VBM, CBM, gap) -- aligned to VBM for a semiconductor
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']

print(f"VBM = {vbm:.4f} eV")
print(f"CBM = {cbm:.4f} eV")
print(f"Band gap = {gap:.4f} eV")

# Align to VBM (semiconductor)
zero = vbm

# Plot
fig, axp = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in ax['segments']:
    for b in range(E.shape[1]):
        axp.plot(x[s], E[s, b] - zero, 'k-', lw=1)

# Draw vertical lines at every tick
for t in ax['tick_x']:
    axp.axvline(t, color='gray', lw=0.5)

# Label ticks
axp.set_xticks(ax['tick_x'])
axp.set_xticklabels(ax['tick_labels'])

# Set limits
axp.set_xlim(x[0], x[-1])
axp.set_ylim(-10, 10)

axp.set_xlabel('k-point')
axp.set_ylabel('Energy (eV)')
axp.set_title('Bi2Se3 band structure (spin-orbit, aligned to VBM)')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print("Saved calculations/000/bands_000.pdf")
