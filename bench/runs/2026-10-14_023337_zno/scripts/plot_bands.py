import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# =============================================================================
# plot_bands.py
# Plots the band structure for calculations/000 (wurtzite ZnO, unperturbed).
# - x-axis, energies, ticks, segments from chem_llm.kpath.plot_axis
# - Aligns to VBM (ZnO is a wide-gap insulator, not a metal)
# - Saves to calculations/000/bands_000.pdf
# =============================================================================

kpath_json = 'calculations/000/kpath.json'
bands_dat = 'calculations/000/000.bands.dat'
pw_out = 'calculations/000/pw.out'

# Get the plot axis (x, energies, ticks, segments) from plot_axis
ax = plot_axis(kpath_json, bands_dat)
x = ax['x']
E = ax['energies']  # shape: (n_kpoints, n_bands), in eV

# Read the SCF output to get n_occupied and Fermi energy
pw = read_pw_out(pw_out)
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']
print(f"n_occupied = {n_occupied}")
if fermi_eV is not None:
    print(f"Fermi energy = {fermi_eV:.4f} eV")
else:
    print("Fermi energy: not reported (insulator with fixed occupations)")

# ZnO is an insulator -> align to VBM
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f"VBM = {vbm:.4f} eV")
print(f"CBM = {cbm:.4f} eV")
print(f"Band gap = {gap:.4f} eV")

# Align energies to VBM
E_shifted = E - vbm

# Plot
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in ax['segments']:
    for b in range(E_shifted.shape[1]):
        plt_ax.plot(x[s], E_shifted[s, b], 'k-', lw=1)

# Draw vertical lines at every tick
for t in ax['tick_x']:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Label the ticks
plt_ax.set_xticks(ax['tick_x'])
plt_ax.set_xticklabels(ax['tick_labels'])

# Set axis limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)
plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV) relative to VBM')
plt_ax.set_title('ZnO (wurtzite) band structure - PBEsol, 92 Ry cutoff')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print("Saved plot to calculations/000/bands_000.pdf")
print("Done.")
