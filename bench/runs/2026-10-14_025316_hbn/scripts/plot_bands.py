import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# ============================================================
# plot_bands.py
# Plot the band structure for calculations/000 (h-BN, mp-629015).
# Aligns to the VBM (h-BN is a clear insulator, band_gap_eV=3.97).
# ============================================================

kpath_json = 'calculations/000/kpath.json'
bands_dat = 'calculations/000/000.bands.dat'
pw_out = 'calculations/000/pw.out'

# Get x-axis, energies, ticks, and line segments from plot_axis (INVARIANT 4)
ax = plot_axis(kpath_json, bands_dat)
x = ax['x']
E = ax['energies']  # shape: (n_kpts, n_bands), energies in eV

# Read SCF output to get n_occupied and Fermi energy (INVARIANT 2)
pw = read_pw_out(pw_out)
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']

print(f'Number of occupied bands: {n_occupied}')
if fermi_eV is not None:
    print(f'Fermi energy: {fermi_eV:.4f} eV')
else:
    print('Fermi energy: not reported (fixed occupations, insulator)')

# Align to VBM (h-BN is an insulator, not a metal)
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']

print(f'VBM: {vbm:.4f} eV')
print(f'CBM: {cbm:.4f} eV')
print(f'Band gap: {gap:.4f} eV')

# Plot each band as one line per segment (never across a discontinuity)
fig, plt_ax = plt.subplots(figsize=(10, 6))

for s in ax['segments']:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], E[s, b] - vbm, 'k-', lw=1)

# Draw vertical lines at every tick
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
plt_ax.set_ylabel('Energy (eV) relative to VBM')
plt_ax.set_title('h-BN Band Structure (PBEsol, mp-629015)')
plt_ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print('Saved calculations/000/bands_000.pdf')
