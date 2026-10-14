import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# ============================================================
# plot_bands.py
# Plot the band structure for calculations/000 (Al, fcc).
# Al is a metal (band_gap_eV = 0.0), so align to the Fermi energy.
# ============================================================

KPATH_JSON = 'calculations/000/kpath.json'
BANDS_DAT = 'calculations/000/000.bands.dat'
PW_OUT = 'calculations/000/pw.out'
OUTPUT_PDF = 'calculations/000/bands_000.pdf'

# Get x-axis, energies, ticks, and line segments from plot_axis
# (checks that bands were computed on kpath.json's k-points)
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']
E = ax['energies']  # shape: (n_kpts, n_bands)

print(f"k-points: {len(x)}, bands: {E.shape[1]}")

# Read pw.out for n_occupied and fermi_energy_eV
pw = read_pw_out(PW_OUT)
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']
print(f"n_occupied = {n_occupied}, fermi_energy_eV = {fermi_eV}")

# Al is a metal, so align to the Fermi energy
zero = fermi_eV
print(f"Aligning to Fermi energy: {zero:.4f} eV")

# Plot each band as one line per segment (never across a discontinuity)
fig, plt_ax = plt.subplots(figsize=(10, 6))
for s in ax['segments']:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], E[s, b] - zero, 'k-', lw=1)

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
plt_ax.set_title('Al fcc band structure (PBEsol, aligned to Fermi energy)')
plt_ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PDF, bbox_inches='tight')
print(f"Saved {OUTPUT_PDF}")

# Print VBM, CBM, and band gap (or Fermi energy for a metal)
edges = band_edges(E, n_occupied)
print(f"VBM = {edges['vbm']:.4f} eV")
print(f"CBM = {edges['cbm']:.4f} eV")
print(f"Band gap = {edges['gap']:.4f} eV")
print(f"Fermi energy = {fermi_eV:.4f} eV")
