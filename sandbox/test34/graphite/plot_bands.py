"""
plot_bands.py - Plot the band structure for calculations/000 (graphite, mp-48).

Graphite is a semimetal (band_gap_eV=0.0 from generate_cif), so we align
to the Fermi energy read from pw.out via chem_llm.qe_output.read_pw_out.

Uses:
  - chem_llm.kpath.plot_axis for x-axis, energies, ticks, and line segments
  - chem_llm.qe_output.read_pw_out for n_occupied and fermi_energy_eV
  - chem_llm.qe_output.band_edges for VBM/CBM/gap (for reference)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# Paths
KPATH_JSON = 'calculations/000/kpath.json'
BANDS_DAT = 'calculations/000/000.bands.dat'
PW_OUT = 'calculations/000/pw.out'
OUTPUT_PDF = 'calculations/000/bands_000.pdf'

# Get x-axis, energies, ticks, and segments from plot_axis
# This also verifies that bands.x computed the same k-points as kpath.json
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']           # path length (array of shape (N_kpts,))
E = ax['energies']    # energies in eV (array of shape (N_kpts, N_bands))
segments = ax['segments']  # list of slice objects, one per continuous segment
tick_x = ax['tick_x']      # list of float x-positions for ticks
tick_labels = ax['tick_labels']  # list of string labels

print(f"Number of k-points: {len(x)}")
print(f"Number of bands: {E.shape[1]}")
print(f"Segments: {len(segments)}")
print(f"Ticks: {list(zip(tick_x, tick_labels))}")

# Read pw.out for n_occupied and fermi energy
pw = read_pw_out(PW_OUT)
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']
print(f"n_occupied = {n_occupied}")
print(f"Fermi energy = {fermi_eV:.4f} eV")

# Graphite is a semimetal -> align to Fermi energy
zero = fermi_eV
print(f"Aligning to Fermi energy: {zero:.4f} eV")

# Also compute VBM/CBM/gap for reference (using n_occupied)
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f"VBM = {vbm:.4f} eV (relative to zero)")
print(f"CBM = {cbm:.4f} eV (relative to zero)")
print(f"Band gap = {gap:.4f} eV")

# Plot
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in segments:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], E[s, b] - zero, 'k-', lw=1)

# Draw vertical lines at every tick
for t in tick_x:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Set x-axis ticks and labels
plt_ax.set_xticks(tick_x)
plt_ax.set_xticklabels(tick_labels)

# Set limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)  # +/-10 eV around Fermi level

# Labels and title
plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title('Graphite (mp-48) band structure\nPBESOL, ecutwfc=82 Ry, smearing=mv, degauss=0.01 Ry')

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=150, bbox_inches='tight')
print(f"\nSaved plot to {OUTPUT_PDF}")

# Print summary
print("\n=== Summary ===")
print(f"Fermi energy: {fermi_eV:.4f} eV")
print(f"VBM (relative to Fermi): {vbm - zero:.4f} eV")
print(f"CBM (relative to Fermi): {cbm - zero:.4f} eV")
print(f"Band gap: {gap:.4f} eV")
print("Graphite is a semimetal: bands cross the Fermi level.")
