import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# Read the band structure data and align to VBM (CdTe is a semiconductor)
kpath_json = 'calculations/000/kpath.json'
bands_dat = 'calculations/000/000.bands.dat'
pw_out = 'calculations/000/pw.out'

# Get x-axis, energies, ticks, and line segments from plot_axis
# This checks that bands were computed on kpath.json's k-points
ax = plot_axis(kpath_json, bands_dat)
x = ax['x']
E = ax['energies']
print(f"Energies shape: {E.shape}")
print(f"Number of k-points: {len(x)}")

# Read pw.out to get n_occupied and Fermi energy
pw = read_pw_out(pw_out)
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']
print(f"n_occupied: {n_occupied}")
print(f"Fermi energy: {fermi_eV} eV")

# CdTe is a semiconductor (gap ~0.59 eV), so align to VBM
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f"VBM: {vbm} eV")
print(f"CBM: {cbm} eV")
print(f"Band gap: {gap} eV")

# Align energies to VBM
E_aligned = E - vbm

# Plot each band as one line per segment (never across a discontinuity)
fig, plt_ax = plt.subplots(figsize=(10, 6))
for s in ax['segments']:
    for b in range(E_aligned.shape[1]):
        plt_ax.plot(x[s], E_aligned[s, b], 'k-', lw=1)

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

# Labels and title
plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title(f'CdTe band structure (VBM-aligned, gap = {gap:.3f} eV)')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print("Saved calculations/000/bands_000.pdf")
