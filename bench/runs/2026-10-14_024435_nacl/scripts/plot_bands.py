import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# =============================================================================
# plot_bands.py
# Plot the band structure for calculations/000 (unperturbed NaCl, Fm-3m #225).
# Aligns energies to the VBM (NaCl is an insulator, gap ~5 eV).
# =============================================================================

kpath_json = 'calculations/000/kpath.json'
bands_dat = 'calculations/000/000.bands.dat'
pw_out = 'calculations/000/pw.out'

# Get x-axis, energies, ticks and line segments from plot_axis
# (checks that bands were computed on kpath.json's k-points)
ax = plot_axis(kpath_json, bands_dat)
x = ax['x']
E = ax['energies']
print(f"Bands: {E.shape[0]} k-points, {E.shape[1]} bands")

# Read SCF output to get n_occupied and Fermi energy
pw = read_pw_out(pw_out)
n_occupied = pw['n_occupied']
print(f"n_occupied = {n_occupied}, n_electrons = {pw['n_electrons']}")

# INVARIANT 2: verify nbnd is sufficient
# nbnd was set to 20 in setup_jobs.py; occupied bands = 8 (16 electrons / 2)
assert E.shape[1] >= n_occupied, f"nbnd too small: {E.shape[1]} < {n_occupied}"

# NaCl is an insulator -> align to VBM
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f"VBM = {vbm:.4f} eV")
print(f"CBM = {cbm:.4f} eV")
print(f"Band gap = {gap:.4f} eV")

# Plot each band as one line per segment (never across a discontinuity)
fig, plt_ax = plt.subplots(figsize=(10, 6))
for s in ax['segments']:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], E[s, b] - vbm, 'k-', lw=1)

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
plt_ax.set_title('NaCl band structure (PBEsol, 88 Ry)')
plt_ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print("Saved calculations/000/bands_000.pdf")
