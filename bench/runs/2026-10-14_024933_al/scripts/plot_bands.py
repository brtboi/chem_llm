import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# =============================================================================
# plot_bands.py
# Plot the band structure for calculations/000 (fcc Al, 1 atom).
# Al is a metal (band_gap_eV=0.0 from generate_cif), so align to the Fermi
# energy read from pw.out, not to a VBM.
# =============================================================================

kpath_json = 'calculations/000/kpath.json'
bands_dat = 'calculations/000/000.bands.dat'
pw_out = 'calculations/000/pw.out'

# --- Read the band data and x-axis from plot_axis (INVARIANT 4) ---
# plot_axis checks that bands.x computed the same k-points as kpath.json.
ax = plot_axis(kpath_json, bands_dat)
x = ax['x']           # path length (A^-1 or similar), shape (N_kpts,)
E = ax['energies']    # energies in eV, shape (N_kpts, n_bands)
segments = ax['segments']   # list of slice objects, one per continuous segment
tick_x = ax['tick_x']       # list of tick positions
tick_labels = ax['tick_labels']  # list of tick labels (e.g. 'GAMMA', 'X', 'U|K')

print(f"k-path: {len(x)} k-points, {E.shape[1]} bands")
print(f"Ticks: {list(zip(tick_x, tick_labels))}")

# --- Read pw.out to get n_occupied and Fermi energy (INVARIANT 2) ---
pw = read_pw_out(pw_out)
n_electrons = pw['n_electrons']
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']
print(f"n_electrons = {n_electrons}, n_occupied = {n_occupied}, E_F = {fermi_eV:.4f} eV")

# Al is a metal: align to the Fermi energy, not the VBM.
zero = fermi_eV
print(f"Aligning to Fermi energy: {zero:.4f} eV")

# --- Plot ---
fig, ax_plot = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in segments:
    for b in range(E.shape[1]):
        ax_plot.plot(x[s], E[s, b] - zero, 'k-', lw=1)

# Draw a vertical line at every tick
for t in tick_x:
    ax_plot.axvline(t, color='gray', lw=0.5)

# Label the ticks
ax_plot.set_xticks(tick_x)
ax_plot.set_xticklabels(tick_labels)

# Set axis limits
ax_plot.set_xlim(x[0], x[-1])
ax_plot.set_ylim(-10, 10)
ax_plot.set_xlabel('k-point')
ax_plot.set_ylabel('Energy (eV)')
ax_plot.set_title('Al band structure (majority spin, aligned to E_F)')

plt.tight_layout()
out_pdf = 'calculations/000/bands_000.pdf'
plt.savefig(out_pdf, dpi=150)
print(f"Saved plot to {out_pdf}")

# --- Print band edges for reference ---
# For a metal, the "gap" is 0 by definition, but print VBM/CBM relative to E_F
edges = band_edges(E, n_occupied)
print(f"VBM (relative to E_F): {edges['vbm'] - zero:.4f} eV")
print(f"CBM (relative to E_F): {edges['cbm'] - zero:.4f} eV")
print(f"Band gap: {edges['gap']:.4f} eV (metal: gap = 0)")
