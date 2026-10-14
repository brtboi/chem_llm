import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# =============================================================================
# plot_bands.py
# Plot the band structure for calculations/000 (SrTiO3, Pm-3m, 5 atoms)
# Aligned to VBM (semiconductor, gap ~1.77 eV from MP)
# =============================================================================

KPATH_JSON = 'calculations/000/kpath.json'
BANDS_DAT = 'calculations/000/000.bands.dat'
PW_OUT = 'calculations/000/pw.out'
OUTPUT_PDF = 'calculations/000/bands_000.pdf'

# --- Get x-axis, energies, ticks, segments from plot_axis (INVARIANT 4) ---
# plot_axis checks that bands were computed on kpath.json's k-points.
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']          # path length (array of shape (n_kpts,))
E = ax['energies']   # energies in eV (array of shape (n_kpts, n_bands))

# --- Read SCF output for n_occupied (INVARIANT 2) ---
pw = read_pw_out(PW_OUT)
n_occupied = pw['n_occupied']
print(f"n_occupied from pw.out: {n_occupied}")
print(f"n_electrons from pw.out: {pw['n_electrons']}")
print(f"Fermi energy from pw.out: {pw['fermi_energy_eV']} eV")

# --- Align to VBM (semiconductor) ---
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f"VBM: {vbm:.4f} eV")
print(f"CBM: {cbm:.4f} eV")
print(f"Band gap: {gap:.4f} eV")

# --- Plot ---
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in ax['segments']:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], E[s, b] - vbm, 'k-', lw=1)

# Draw vertical line at every tick
for t in ax['tick_x']:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Label ticks from plot_axis (they already read 'A|L' at discontinuities)
plt_ax.set_xticks(ax['tick_x'])
plt_ax.set_xticklabels(ax['tick_labels'])

# Set x-limits to full path
plt_ax.set_xlim(x[0], x[-1])

# Restrict y-axis to about +/-10 eV
plt_ax.set_ylim(-10, 10)

# Labels and title
plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title(f'SrTiO3 band structure (PBEsol, 5 atoms, Pm-3m)\nGap = {gap:.3f} eV (VBM-aligned)')

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=150, bbox_inches='tight')
print(f"Saved band structure plot to {OUTPUT_PDF}")
