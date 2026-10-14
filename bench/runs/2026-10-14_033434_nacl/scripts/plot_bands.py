import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# ============================================================
# STEP 7: Plot the band structure for calculations/000
# ============================================================
# NaCl is an insulator (gap ~5 eV), so align to the VBM.

# Get x-axis, energies, ticks, and line segments from plot_axis
# (INVARIANT 4: it checks that bands were computed on kpath.json's k-points)
ax = plot_axis('calculations/000/kpath.json', 'calculations/000/000.bands.dat')
x = ax['x']
E = ax['energies']

# Read pw.out to get n_occupied (INVARIANT 2)
pw = read_pw_out('calculations/000/pw.out')
n_occupied = pw['n_occupied']
print(f"n_occupied from pw.out: {n_occupied}")
print(f"n_electrons from pw.out: {pw['n_electrons']}")

# Align to VBM (insulator)
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f"VBM: {vbm:.4f} eV")
print(f"CBM: {cbm:.4f} eV")
print(f"Band gap: {gap:.4f} eV")

# Plot each band as one line per segment (never across a discontinuity)
fig, ax_plot = plt.subplots(figsize=(10, 6))
for s in ax['segments']:
    for b in range(E.shape[1]):
        ax_plot.plot(x[s], E[s, b] - vbm, 'k-', lw=1)

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
ax_plot.set_ylabel('Energy (eV)')
ax_plot.set_title('NaCl band structure (PBEsol, aligned to VBM)')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf')
print("Saved calculations/000/bands_000.pdf")
