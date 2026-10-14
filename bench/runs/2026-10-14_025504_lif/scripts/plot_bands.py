import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# =============================================================================
# plot_bands.py
# Plots the band structure for calculations/000 (LiF, rock salt, Fm-3m).
# - x-axis, energies, ticks, segments from chem_llm.kpath.plot_axis
# - Align to VBM (LiF is a clear insulator, gap ~8.7 eV)
# - Save to calculations/000/bands_000.pdf
# =============================================================================

kpath_json = 'calculations/000/kpath.json'
bands_dat = 'calculations/000/000.bands.dat'
pw_out = 'calculations/000/pw.out'

# Get x-axis, energies, ticks, segments from plot_axis (INVARIANT 4)
ax = plot_axis(kpath_json, bands_dat)
x = ax['x']          # path length (A^-1)
E = ax['energies']   # shape (n_kpts, n_bands), in eV

# Read SCF output for n_occupied and Fermi energy (INVARIANT 2)
pw = read_pw_out(pw_out)
n_occ = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']

# For a non-smearing SCF run (occupations='fixed'), QE does not print a
# Fermi energy line, so fermi_eV is None. Guard the print accordingly.
if fermi_eV is not None:
    print(f'n_occupied = {n_occ}, fermi_energy = {fermi_eV:.4f} eV')
else:
    print(f'n_occupied = {n_occ}, fermi_energy = None (no smearing, fixed occupations)')

# Align to VBM (insulator)
edges = band_edges(E, n_occ)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f'VBM = {vbm:.4f} eV, CBM = {cbm:.4f} eV, gap = {gap:.4f} eV')

# Plot: one line per segment, never across a discontinuity
fig, axp = plt.subplots(figsize=(10, 6))
for s in ax['segments']:
    for b in range(E.shape[1]):
        axp.plot(x[s], E[s, b] - vbm, 'k-', lw=1)

# Vertical lines at every tick
for t in ax['tick_x']:
    axp.axvline(t, color='gray', lw=0.5)

# Tick labels from plot_axis (already 'A|L' at discontinuities)
axp.set_xticks(ax['tick_x'])
axp.set_xticklabels(ax['tick_labels'])
axp.set_xlim(x[0], x[-1])
axp.set_ylim(-10, 10)
axp.set_xlabel('k-path (A$^{-1}$)')
axp.set_ylabel('Energy (eV)')
axp.set_title('LiF band structure (PBEsol, VBM-aligned)')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print('Saved calculations/000/bands_000.pdf')
