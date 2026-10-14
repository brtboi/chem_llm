import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# ============================================================
# plot_bands.py
# Plot the band structure for calculations/000 (wurtzite ZnO).
# Aligns to VBM (insulator), saves to calculations/000/bands_000.pdf.
# ============================================================

kpath_json = 'calculations/000/kpath.json'
bands_dat = 'calculations/000/000.bands.dat'
pw_out = 'calculations/000/pw.out'

# Get x-axis, energies, ticks, segments from plot_axis (INVARIANT 4)
ax = plot_axis(kpath_json, bands_dat)
x = ax['x']
E = ax['energies']  # shape (n_kpts, n_bands), in eV
print(f"Energies shape: {E.shape}")
print(f"x range: {x[0]:.4f} to {x[-1]:.4f}")
print(f"Ticks: {ax['tick_labels']}")

# Read pw.out for n_occupied (INVARIANT 2)
pw = read_pw_out(pw_out)
n_occupied = pw['n_occupied']
print(f"n_occupied from pw.out: {n_occupied}")
print(f"n_electrons from pw.out: {pw['n_electrons']}")

# Align to VBM (insulator, not metal)
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f"VBM: {vbm:.4f} eV")
print(f"CBM: {cbm:.4f} eV")
print(f"Band gap: {gap:.4f} eV")

# Plot
fig, ax_plot = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in ax['segments']:
    for b in range(E.shape[1]):
        ax_plot.plot(x[s], E[s, b] - vbm, 'k-', lw=1)

# Draw vertical lines at every tick
for t in ax['tick_x']:
    ax_plot.axvline(t, color='gray', lw=0.5)

# Label ticks
ax_plot.set_xticks(ax['tick_x'])
ax_plot.set_xticklabels(ax['tick_labels'])

# Set limits
ax_plot.set_xlim(x[0], x[-1])
ax_plot.set_ylim(-10, 10)
ax_plot.set_xlabel('k-point')
ax_plot.set_ylabel('Energy (eV) relative to VBM')
ax_plot.set_title('Wurtzite ZnO band structure (PBEsol, SCF + NSCF)')
ax_plot.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print("Saved calculations/000/bands_000.pdf")
