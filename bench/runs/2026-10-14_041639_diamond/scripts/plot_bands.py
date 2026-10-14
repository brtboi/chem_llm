import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# =============================================================================
# plot_bands.py
# Plots the band structure for calculations/000 (diamond C, unperturbed).
# - x-axis, energies, ticks, segments from chem_llm.kpath.plot_axis
# - Align to VBM (diamond C is an insulator, gap ~4.1 eV)
# - n_occupied from chem_llm.qe_output.read_pw_out (INVARIANT 2)
# =============================================================================

kpath_json = 'calculations/000/kpath.json'
bands_dat = 'calculations/000/000.bands.dat'
pw_out = 'calculations/000/pw.out'

# Get x-axis, energies, ticks, segments from plot_axis (INVARIANT 4)
ax = plot_axis(kpath_json, bands_dat)
x = ax['x']
E = ax['energies']  # shape (nkpts, nbands), in eV

# Read pw.out to get n_occupied (INVARIANT 2)
pw = read_pw_out(pw_out)
n_occupied = pw['n_occupied']
print(f"n_occupied from pw.out: {n_occupied}")
print(f"Fermi energy from pw.out: {pw['fermi_energy_eV']} eV")

# Diamond C is an insulator -> align to VBM
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f"VBM = {vbm:.4f} eV")
print(f"CBM = {cbm:.4f} eV")
print(f"Band gap = {gap:.4f} eV")

# Plot each band as one line per segment (never across a discontinuity)
fig, axp = plt.subplots(figsize=(10, 6))
for s in ax['segments']:
    for b in range(E.shape[1]):
        axp.plot(x[s], E[s, b] - vbm, 'k-', lw=1)

# Draw vertical lines at every tick
for t in ax['tick_x']:
    axp.axvline(t, color='gray', lw=0.5)

# Label ticks from plot_axis
axp.set_xticks(ax['tick_x'])
axp.set_xticklabels(ax['tick_labels'])

# Set x-limits to full path
axp.set_xlim(x[0], x[-1])

# Restrict y-axis to about +/-10 eV
axp.set_ylim(-10, 10)

axp.set_xlabel('k-point')
axp.set_ylabel('Energy (eV) relative to VBM')
axp.set_title('Diamond C band structure (PBEsol, 82 Ry, 13x13x13 k-mesh)')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print("Saved calculations/000/bands_000.pdf")
