import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# =============================================================================
# plot_bands.py
# Plots the band structure for calculations/000 (GaAs, zincblende).
# Uses chem_llm.kpath.plot_axis for the x-axis, ticks, and line segments
# (INVARIANT 4), and chem_llm.qe_output for reading pw.out and computing
# band edges (INVARIANT 2).
# =============================================================================

kpath_json = 'calculations/000/kpath.json'
bands_dat = 'calculations/000/000.bands.dat'
pw_out = 'calculations/000/pw.out'

# Get x-axis, energies, ticks, and line segments from plot_axis
ax = plot_axis(kpath_json, bands_dat)
x = ax['x']
E = ax['energies']  # shape: (n_kpts, n_bands), energies in eV

# Read pw.out to get the number of occupied bands and Fermi energy
pw = read_pw_out(pw_out)
n_occupied = pw['n_occupied']
print(f"Number of occupied bands: {n_occupied}")

# Fermi energy may be None for a non-smearing run; handle gracefully
if pw['fermi_energy_eV'] is not None:
    print(f"Fermi energy: {pw['fermi_energy_eV']:.4f} eV")
else:
    print("Fermi energy: not available (no smearing)")

# GaAs is a semiconductor (band_gap_eV=0.184 from MP), so align to VBM
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f"VBM: {vbm:.4f} eV")
print(f"CBM: {cbm:.4f} eV")
print(f"Band gap: {gap:.4f} eV")

# Align energies to VBM (set VBM = 0)
E_shifted = E - vbm

# Plot each band as one line per segment (never across a discontinuity)
fig, ax_plot = plt.subplots(figsize=(10, 6))
for s in ax['segments']:
    for b in range(E_shifted.shape[1]):
        ax_plot.plot(x[s], E_shifted[s, b], 'k-', lw=1)

# Draw a vertical line at every tick
for t in ax['tick_x']:
    ax_plot.axvline(t, color='gray', lw=0.5)

# Label the ticks from plot_axis's ticks
ax_plot.set_xticks(ax['tick_x'])
ax_plot.set_xticklabels(ax['tick_labels'])

# Set x-limits to the full path
ax_plot.set_xlim(x[0], x[-1])

# Restrict y-axis to about +/-10 eV
ax_plot.set_ylim(-10, 10)

ax_plot.set_xlabel('k-point')
ax_plot.set_ylabel('Energy (eV) relative to VBM')
ax_plot.set_title('GaAs Band Structure (PBEsol, scalar relativistic)')
ax_plot.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print("Saved band structure plot to calculations/000/bands_000.pdf")
