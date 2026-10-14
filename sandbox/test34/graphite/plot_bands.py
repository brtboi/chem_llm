"""
plot_bands.py -- Plot the band structure for calculations/000 (graphite, C4).

Uses:
  - chem_llm.kpath.plot_axis for x-axis, ticks, segments (INVARIANT 4)
  - chem_llm.qe_output.read_pw_out for n_occupied and fermi_energy_eV (INVARIANT 2)
  - chem_llm.qe_output.band_edges for VBM/CBM/gap

Graphite is a semimetal (band_gap_eV = 0.0 from Materials Project), so we
align to the Fermi energy from pw.out rather than the VBM.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# --- Load band data ---
ax = plot_axis('calculations/000/kpath.json', 'calculations/000/000.bands.dat')
x = ax['x']          # path length (arbitrary units)
E = ax['energies']   # shape (nkpts, nbands), in eV

# --- Read SCF output for electron count and Fermi energy ---
pw = read_pw_out('calculations/000/pw.out')
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']

print(f"n_occupied = {n_occupied}")
print(f"Fermi energy = {fermi_eV:.4f} eV")

# --- Band edges (for reference) ---
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']

print(f"VBM = {vbm:.4f} eV")
print(f"CBM = {cbm:.4f} eV")
print(f"Band gap = {gap:.4f} eV")

# --- Align to Fermi energy (semimetal) ---
zero = fermi_eV
E_shifted = E - zero

# --- Plot ---
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Draw each band as one line per segment (never across a discontinuity)
for s in ax['segments']:
    for b in range(E_shifted.shape[1]):
        plt_ax.plot(x[s], E_shifted[s, b], 'k-', lw=1)

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
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title('Graphite C4 band structure (PBEsol, scalar-relativistic)\n' +
                 f'Fermi energy aligned to 0 eV, gap = {gap:.4f} eV')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print("Saved calculations/000/bands_000.pdf")
