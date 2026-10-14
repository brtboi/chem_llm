import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# =============================================================================
# plot_bands.py
# Plot the band structure for calculations/000 (hBN, P6_3/mmc).
# Aligns to the VBM (insulator, band_gap_eV ~ 3.97 from Materials Project).
# =============================================================================

CALC_DIR = 'calculations/000'
KPATH_JSON = f'{CALC_DIR}/kpath.json'
BANDS_DAT = f'{CALC_DIR}/000.bands.dat'
PW_OUT = f'{CALC_DIR}/pw.out'
OUTPUT_PDF = f'{CALC_DIR}/bands_000.pdf'

# --- Get x-axis, energies, ticks, segments from plot_axis (INVARIANT 4) ------
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']          # path length (A^-1)
E = ax['energies']   # shape (n_kpts, n_bands), in eV
segments = ax['segments']
tick_x = ax['tick_x']
tick_labels = ax['tick_labels']

# --- Read SCF output for electron count and Fermi energy (INVARIANT 2) -------
pw = read_pw_out(PW_OUT)
n_occupied = pw['n_occupied']  # number of occupied bands (spin-unpolarized)
fermi_eV = pw['fermi_energy_eV']

print(f"Number of electrons: {pw['n_electrons']}")
print(f"Number of occupied bands: {n_occupied}")
if fermi_eV is not None:
    print(f"Fermi energy: {fermi_eV:.4f} eV")
else:
    print("Fermi energy: not reported (insulator with fixed occupations)")

# --- Align to VBM (insulator) -------------------------------------------------
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']

print(f"VBM: {vbm:.4f} eV")
print(f"CBM: {cbm:.4f} eV")
print(f"Band gap: {gap:.4f} eV")

# --- Plot ---------------------------------------------------------------------
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in segments:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], E[s, b] - vbm, 'k-', lw=1)

# Draw vertical lines at every tick
for t in tick_x:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Label ticks
plt_ax.set_xticks(tick_x)
plt_ax.set_xticklabels(tick_labels)

# Set axis limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)

# Labels and title
plt_ax.set_xlabel('k-point path (A$^{-1}$)')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title('hBN band structure (PBEsol, aligned to VBM)')

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=150)
print(f"Saved plot to {OUTPUT_PDF}")
