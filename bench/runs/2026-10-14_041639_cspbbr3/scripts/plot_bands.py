import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# =============================================================================
# plot_bands.py
# Plots the band structure for calculations/000 (CsPbBr3, Pm-3m).
# - x-axis, ticks, line segments from chem_llm.kpath.plot_axis (INVARIANT 4)
# - energies aligned to VBM (semiconductor, not a metal)
# - VBM, CBM, and band gap printed
# =============================================================================

KPATH_JSON = 'calculations/000/kpath.json'
BANDS_DAT = 'calculations/000/000.bands.dat'
PW_OUT = 'calculations/000/pw.out'
OUT_PDF = 'calculations/000/bands_000.pdf'

# --- Get x-axis, energies, ticks, segments from plot_axis (INVARIANT 4) ---
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']          # path length (arbitrary units)
E = ax['energies']   # shape (n_kpts, n_bands), in eV

# --- Read SCF output for n_occupied (INVARIANT 2) ---
pw = read_pw_out(PW_OUT)
n_occ = pw['n_occupied']
print(f'Number of electrons: {pw["n_electrons"]}')
print(f'Number of occupied bands: {n_occ}')
print(f'Fermi energy: {pw["fermi_energy_eV"]} eV')

# --- Align to VBM (CsPbBr3 is a semiconductor, gap ~1.78 eV) ---
edges = band_edges(E, n_occ)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f'VBM: {vbm:.4f} eV')
print(f'CBM: {cbm:.4f} eV')
print(f'Band gap: {gap:.4f} eV')

# --- Plot ---
fig, axp = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in ax['segments']:
    for b in range(E.shape[1]):
        axp.plot(x[s], E[s, b] - vbm, 'k-', lw=0.8)

# Draw vertical line at every tick
for t in ax['tick_x']:
    axp.axvline(t, color='gray', lw=0.5)

# Label ticks from plot_axis (they already read 'A|L' at discontinuities)
axp.set_xticks(ax['tick_x'])
axp.set_xticklabels(ax['tick_labels'])

# Set x-limits to full path
axp.set_xlim(x[0], x[-1])

# Restrict y-axis to about +/-10 eV around VBM
axp.set_ylim(-10, 10)

axp.set_xlabel('k-point')
axp.set_ylabel('Energy (eV)')
axp.set_title('CsPbBr3 band structure (PBEsol, scalar-relativistic)\n' +
              f'Gap = {gap:.3f} eV (VBM at 0 eV)')

plt.tight_layout()
plt.savefig(OUT_PDF, dpi=150)
print(f'Saved plot to {OUT_PDF}')
