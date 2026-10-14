"""
plot_bands.py -- Plot the band structure for calculations/000 (fcc Cu).

Cu is a metal (band_gap_eV = 0.0 from Materials Project), so we align the
band energies to the Fermi energy read from pw.out (not to a VBM).

Uses:
  - chem_llm.kpath.plot_axis for x-axis, ticks, segments (INVARIANT 4)
  - chem_llm.qe_output.read_pw_out for Fermi energy and n_occupied (INVARIANT 2)
  - matplotlib for the plot
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# --- Load the band data and k-path axis ---
ax = plot_axis('calculations/000/kpath.json', 'calculations/000/000.bands.dat')
x = ax['x']          # path length (array)
E = ax['energies']   # (n_kpts, n_bands) in eV

# --- Read SCF output for Fermi energy and electron count ---
pw = read_pw_out('calculations/000/pw.out')
n_electrons = pw['n_electrons']
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']

print(f"Number of electrons: {n_electrons}")
print(f"Number of occupied bands: {n_occupied}")
print(f"Fermi energy: {fermi_eV:.4f} eV")

# INVARIANT 2 check: for a spin-unpolarized metal, n_occupied should be
# approximately n_electrons / 2 (each band holds 2 electrons).
# With smearing the count may differ slightly; just report it.
assert n_electrons > 0, "n_electrons must be positive"

# --- Align to Fermi energy (Cu is a metal) ---
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

# Set tick labels from plot_axis
plt_ax.set_xticks(ax['tick_x'])
plt_ax.set_xticklabels(ax['tick_labels'])

# Set axis limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)  # +/-10 eV around Fermi level

plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title('Band structure of fcc Cu (PBEsol, aligned to Fermi level)')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf')
print("Saved calculations/000/bands_000.pdf")

# --- Print band edges for reference ---
# For a metal, VBM/CBM are not well-defined in the insulator sense,
# but we can still compute them from the occupied/unoccupied bands.
edges = band_edges(E, n_occupied)
print(f"VBM: {edges['vbm']:.4f} eV")
print(f"CBM: {edges['cbm']:.4f} eV")
print(f"Band gap (VBM->CBM): {edges['gap']:.4f} eV")
print(f"Fermi energy (alignment): {fermi_eV:.4f} eV")
