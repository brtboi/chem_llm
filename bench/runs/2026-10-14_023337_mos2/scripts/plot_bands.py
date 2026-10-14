"""
plot_bands.py: Plot the band structure for calculations/000 (2H MoS2, PBEsol).

Uses chem_llm.kpath.plot_axis for the x-axis, ticks, and line segments (INVARIANT 4).
Uses chem_llm.qe_output.read_pw_out for n_occupied and Fermi energy (INVARIANT 2).
Aligns to VBM (semiconductor, not a metal).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# Read the band data and x-axis from plot_axis (checks k-points match kpath.json)
ax = plot_axis('calculations/000/kpath.json', 'calculations/000/000.bands.dat')
x = ax['x']          # path length (arbitrary units)
E = ax['energies']   # shape (n_kpoints, n_bands), in eV

# Read SCF output for n_occupied and Fermi energy
pw = read_pw_out('calculations/000/pw.out')
n_occ = pw['n_occupied']
print(f"n_occupied = {n_occ}")
print(f"Fermi energy = {pw['fermi_energy_eV']} eV")

# Align to VBM (semiconductor)
edges = band_edges(E, n_occ)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f"VBM = {vbm:.4f} eV")
print(f"CBM = {cbm:.4f} eV")
print(f"Band gap = {gap:.4f} eV")

# Shift energies so VBM is at 0
e_zero = E - vbm

# Plot
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Draw each band as one line per segment (no line across a discontinuity)
for s in ax['segments']:
    for b in range(e_zero.shape[1]):
        plt_ax.plot(x[s], e_zero[s, b], 'k-', lw=1)

# Draw vertical lines at every tick
for t in ax['tick_x']:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Set x-axis ticks and labels
plt_ax.set_xticks(ax['tick_x'])
plt_ax.set_xticklabels(ax['tick_labels'])

# Set limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)

# Labels and title
plt_ax.set_xlabel('k-path')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title('2H MoS2 band structure (PBEsol, scalar relativistic)')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print("Saved calculations/000/bands_000.pdf")
