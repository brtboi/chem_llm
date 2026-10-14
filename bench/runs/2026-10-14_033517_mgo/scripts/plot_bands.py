import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# =============================================================================
# plot_bands.py
# Plots the band structure for calculations/000 (MgO, unperturbed base).
# - x-axis, energies, ticks, segments from chem_llm.kpath.plot_axis
# - Aligns to VBM (MgO is an insulator, band_gap ~ 4.4 eV)
# - Saves calculations/000/bands_000.pdf
# =============================================================================

CALC_DIR = 'calculations/000'
KPATH_JSON = f'{CALC_DIR}/kpath.json'
BANDS_DAT = f'{CALC_DIR}/000.bands.dat'
PW_OUT = f'{CALC_DIR}/pw.out'
OUT_PDF = f'{CALC_DIR}/bands_000.pdf'

# Get x-axis, energies, ticks, segments from plot_axis
# It checks that bands were computed on kpath.json's k-points
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']
E = ax['energies']  # shape: (nkpts, nbands), energies in eV

# Read pw.out for n_occupied and fermi_energy_eV (INVARIANT 2)
pw = read_pw_out(PW_OUT)
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']
print(f"n_occupied = {n_occupied}")
print(f"fermi_energy_eV = {fermi_eV}")

# Align to VBM (insulator)
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f"VBM = {vbm:.4f} eV")
print(f"CBM = {cbm:.4f} eV")
print(f"Band gap = {gap:.4f} eV")

# Plot each band as one line per segment (never across a discontinuity)
fig, ax_plot = plt.subplots(figsize=(10, 6))

for s in ax['segments']:
    for b in range(E.shape[1]):
        ax_plot.plot(x[s], E[s, b] - vbm, 'k-', lw=1)

# Draw vertical line at every tick
for t in ax['tick_x']:
    ax_plot.axvline(t, color='gray', lw=0.5)

# Label ticks from plot_axis's ticks
ax_plot.set_xticks(ax['tick_x'])
ax_plot.set_xticklabels(ax['tick_labels'])

# Set x-limits to full path
ax_plot.set_xlim(x[0], x[-1])

# Restrict y-axis to about +/-10 eV
ax_plot.set_ylim(-10, 10)

ax_plot.set_xlabel('k-point')
ax_plot.set_ylabel('Energy (eV) relative to VBM')
ax_plot.set_title(f'MgO band structure (VBM-aligned, gap = {gap:.2f} eV)')

plt.tight_layout()
plt.savefig(OUT_PDF, dpi=150)
print(f"Saved {OUT_PDF}")
