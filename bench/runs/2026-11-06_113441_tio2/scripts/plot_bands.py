import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# =============================================================================
# plot_bands.py
# Plots the band structure for calculations/000 (rutile TiO2, mp-390).
# - x-axis, energies, ticks, segments from chem_llm.kpath.plot_axis
# - Align to VBM (semiconductor, gap ~2.06 eV from MP)
# - One line per segment (no line across discontinuities)
# - Vertical line at every tick, labels from plot_axis
# - y-axis restricted to +/-10 eV
# =============================================================================

CALC_DIR = 'calculations/000'
KPATH_JSON = f'{CALC_DIR}/kpath.json'
BANDS_DAT = f'{CALC_DIR}/000.bands.dat'
PW_OUT = f'{CALC_DIR}/pw.out'
OUT_PDF = f'{CALC_DIR}/bands_000.pdf'

# Get x-axis, energies, ticks, segments from plot_axis
# This also verifies that bands.x computed the same k-points as kpath.json
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']          # path length (Angstrom^-1 or similar)
E = ax['energies']   # shape (n_kpts, n_bands), in eV

# Read SCF output to get n_occupied and Fermi energy
pw = read_pw_out(PW_OUT)
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']

print(f"n_occupied = {n_occupied}")
if fermi_eV is not None:
    print(f"Fermi energy = {fermi_eV:.4f} eV")
else:
    print("Fermi energy = None (fixed occupations, no smearing)")

# Align to VBM (semiconductor)
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']

print(f"VBM = {vbm:.4f} eV")
print(f"CBM = {cbm:.4f} eV")
print(f"Band gap = {gap:.4f} eV")

# Shift energies so VBM is at 0
eV_shifted = E - vbm

# Plot
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Draw each band as one line per segment (never across a discontinuity)
for s in ax['segments']:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], eV_shifted[s, b], 'k-', lw=0.8)

# Draw vertical line at every tick
for t in ax['tick_x']:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Set x-ticks and labels
plt_ax.set_xticks(ax['tick_x'])
plt_ax.set_xticklabels(ax['tick_labels'])

# Set limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)

# Labels and title
plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title('Rutile TiO$_2$ band structure (PBEsol, aligned to VBM)')

plt.tight_layout()
plt.savefig(OUT_PDF, dpi=150)
print(f"Saved plot to {OUT_PDF}")
