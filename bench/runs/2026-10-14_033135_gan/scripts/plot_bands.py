import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# =============================================================================
# plot_bands.py
# Plot the band structure for calculations/000 (wurtzite GaN, P6_3mc).
#
# - x-axis, energies, ticks, and line segments come from chem_llm.kpath.plot_axis
#   (INVARIANT 4) -- it verifies that bands.x computed the same k-points as
#   kpath.json.
# - Energies are aligned to the VBM (GaN is a semiconductor, not a metal).
# - n_occupied is read from pw.out via chem_llm.qe_output (INVARIANT 2).
# - Each band is drawn as one line per segment (no line across a discontinuity).
# - Vertical lines at every tick; tick labels from plot_axis (already 'A|L' at
#   discontinuities).
# - y-axis restricted to +/-10 eV around the VBM.
# =============================================================================

kpath_json = 'calculations/000/kpath.json'
bands_dat = 'calculations/000/000.bands.dat'
pw_out = 'calculations/000/pw.out'

# Get the x-axis, energies, ticks, and line segments from plot_axis.
# This function checks that the bands were computed on kpath.json's k-points.
ax = plot_axis(kpath_json, bands_dat)
x = ax['x']          # path length (dimensionless, in units of the path)
E = ax['energies']   # shape (n_kpts, n_bands), in eV
segments = ax['segments']   # list of slice objects, one per continuous segment
tick_x = ax['tick_x']       # list of x positions for ticks
tick_labels = ax['tick_labels']  # list of tick labels

# Read the SCF output to get the number of occupied bands (INVARIANT 2).
pw = read_pw_out(pw_out)
n_occupied = pw['n_occupied']
print(f"n_occupied from pw.out: {n_occupied}")
print(f"n_electrons from pw.out: {pw['n_electrons']}")

# Align energies to the VBM (GaN is a semiconductor).
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f"VBM: {vbm:.4f} eV")
print(f"CBM: {cbm:.4f} eV")
print(f"Band gap: {gap:.4f} eV")

# Shift all energies so the VBM is at 0 eV.
E_shifted = E - vbm

# Plot each band as one line per segment (never across a discontinuity).
fig, plt_ax = plt.subplots(figsize=(10, 6))
for s in segments:
    for b in range(E_shifted.shape[1]):
        plt_ax.plot(x[s], E_shifted[s, b], 'k-', lw=1)

# Draw a vertical line at every tick.
for t in tick_x:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Set x-axis ticks and labels from plot_axis.
plt_ax.set_xticks(tick_x)
plt_ax.set_xticklabels(tick_labels)

# Set x-limits to the full path.
plt_ax.set_xlim(x[0], x[-1])

# Restrict y-axis to +/-10 eV around the VBM.
plt_ax.set_ylim(-10, 10)

# Labels and title.
plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title('Wurtzite GaN band structure (PBEsol, VBM = 0 eV)')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print("Saved band structure plot to calculations/000/bands_000.pdf")
