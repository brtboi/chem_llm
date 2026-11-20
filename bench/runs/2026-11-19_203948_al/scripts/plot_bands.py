"""
STEP 7: Plot the band structure for calculations/000 (fcc Al, mp-134).

Al is a metal (band_gap_eV = 0.0 from MP), so we align energies to the Fermi
energy read from pw.out via chem_llm.dft.qe_output.read_pw_out.

The x-axis, energies, ticks and line segments come from
chem_llm.dft.kpath.plot_axis (INVARIANT 4).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.dft.kpath import plot_axis
from chem_llm.dft.qe_output import read_pw_out, band_edges

# --- Read the band data and the SCF output ---
kpath_json = 'calculations/000/kpath.json'
bands_dat = 'calculations/000/000.bands.dat'
pw_out = 'calculations/000/pw.out'

# plot_axis checks that bands.dat was computed on kpath.json's k-points
ax = plot_axis(kpath_json, bands_dat)
x = ax['x']          # path length (A^-1 or similar)
E = ax['energies']   # shape (n_kpts, n_bands), in eV

# Read SCF output for Fermi energy and electron count
pw = read_pw_out(pw_out)
n_electrons = pw['n_electrons']
fermi_eV = pw['fermi_energy_eV']
print(f"Number of electrons: {n_electrons}")
print(f"Fermi energy: {fermi_eV:.4f} eV")

# Al is a metal -> align to Fermi energy
zero = fermi_eV

# For reference, also compute VBM/CBM (though for a metal the gap is 0)
edges = band_edges(E, pw['n_occupied'])
print(f"VBM: {edges['vbm']:.4f} eV")
print(f"CBM: {edges['cbm']:.4f} eV")
print(f"Band gap: {edges['gap']:.4f} eV")
print(f"Aligned to Fermi energy: {fermi_eV:.4f} eV")

# --- Plot ---
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in ax['segments']:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], E[s, b] - zero, 'k-', lw=1)

# Draw a vertical line at every tick
for t in ax['tick_x']:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Label the ticks from plot_axis's ticks
plt_ax.set_xticks(ax['tick_x'])
plt_ax.set_xticklabels(ax['tick_labels'])

# Set x-limits to the full path
plt_ax.set_xlim(x[0], x[-1])

# Restrict y-axis to about +/-10 eV
plt_ax.set_ylim(-10, 10)

plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title('Al (mp-134) band structure -- aligned to Fermi energy')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print("Saved calculations/000/bands_000.pdf")
