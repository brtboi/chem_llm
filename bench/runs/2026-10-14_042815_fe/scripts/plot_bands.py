import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# =============================================================================
# plot_bands.py
# Plots the band structure for calculations/000 (bcc Fe, spin-polarized).
# Aligns to the Fermi energy (metal).
# =============================================================================

CALC_DIR = 'calculations/000'
KPATH_JSON = f'{CALC_DIR}/kpath.json'
BANDS_DAT = f'{CALC_DIR}/000.bands.dat'
PW_OUT = f'{CALC_DIR}/pw.out'
OUTPUT_PDF = f'{CALC_DIR}/bands_000.pdf'

# --- Get x-axis, energies, ticks, segments from plot_axis (INVARIANT 4) ------
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']           # path length (arbitrary units)
E = ax['energies']    # shape (n_kpts, n_bands), in eV
segments = ax['segments']  # list of slice objects

# --- Read SCF output for Fermi energy and n_occupied (INVARIANT 2) -----------
pw = read_pw_out(PW_OUT)
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']

print(f'Number of electrons: {pw["n_electrons"]}')
print(f'Number of occupied bands: {n_occupied}')
print(f'Fermi energy: {fermi_eV:.4f} eV')
print(f'Spin-orbit: {pw["spin_orbit"]}')
print(f'Total magnetization: {pw["total_magnetization"]}')

# --- Align to Fermi energy (metal) -------------------------------------------
zero = fermi_eV

# --- Compute band edges for reporting ----------------------------------------
edges = band_edges(E, n_occupied)
print(f'\nBand edges (relative to VBM):')
print(f'  VBM: {edges["vbm"]:.4f} eV')
print(f'  CBM: {edges["cbm"]:.4f} eV')
print(f'  Gap: {edges["gap"]:.4f} eV')
print(f'\nFermi energy (alignment point): {fermi_eV:.4f} eV')

# --- Plot --------------------------------------------------------------------
fig, ax_plot = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in segments:
    for b in range(E.shape[1]):
        ax_plot.plot(x[s], E[s, b] - zero, 'k-', lw=0.8)

# Draw vertical lines at every tick
for t in ax['tick_x']:
    ax_plot.axvline(t, color='gray', lw=0.5)

# Label ticks from plot_axis (they already read 'A|L' at discontinuities)
ax_plot.set_xticks(ax['tick_x'])
ax_plot.set_xticklabels(ax['tick_labels'])

# Set x-limits to full path
ax_plot.set_xlim(x[0], x[-1])

# Restrict y-axis to about +/-10 eV
ax_plot.set_ylim(-10, 10)

# Labels and title
ax_plot.set_xlabel('k-point path')
ax_plot.set_ylabel('Energy (eV)')
ax_plot.set_title('bcc Fe band structure (majority spin, aligned to Fermi energy)')

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=150)
print(f'\nSaved plot to {OUTPUT_PDF}')
