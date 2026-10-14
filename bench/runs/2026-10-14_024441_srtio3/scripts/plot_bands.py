import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# =============================================================================
# plot_bands.py
# Plot the band structure for calculations/000 (SrTiO3, Pm-3m, semiconductor)
# Aligned to the VBM (not a metal, so no Fermi-energy alignment needed)
# =============================================================================

KPATH_JSON = 'calculations/000/kpath.json'
BANDS_DAT  = 'calculations/000/000.bands.dat'
PW_OUT     = 'calculations/000/pw.out'
OUT_PDF    = 'calculations/000/bands_000.pdf'

# --- Read the SCF output for electron count and Fermi energy ---
pw = read_pw_out(PW_OUT)
n_occupied = pw['n_occupied']  # number of occupied bands (spin-unpolarized: n_electrons/2)
fermi_eV   = pw['fermi_energy_eV']
print(f'Number of electrons: {pw["n_electrons"]}')
print(f'Number of occupied bands: {n_occupied}')
if fermi_eV is not None:
    print(f'Fermi energy: {fermi_eV:.4f} eV')
else:
    print('Fermi energy: not reported (semiconductor, fixed occupations)')

# INVARIANT 2: n_occupied must be consistent with the electron count
# For spin-unpolarized: n_occupied = n_electrons / 2
assert n_occupied == pw['n_electrons'] / 2, \
    f'n_occupied={n_occupied} != n_electrons/2={pw["n_electrons"]/2}'

# --- Get the x-axis, energies, ticks and segments from plot_axis ---
# plot_axis checks that bands.dat was computed on the same k-points as kpath.json
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x  = ax['x']          # path length (arbitrary units)
E  = ax['energies']   # shape (n_kpoints, n_bands), in eV
tick_x       = ax['tick_x']
tick_labels  = ax['tick_labels']
segments     = ax['segments']  # list of slice objects, one per continuous segment

print(f'k-points: {len(x)}, bands: {E.shape[1]}')
print(f'Ticks: {list(zip(tick_x, tick_labels))}')

# --- Align energies to the VBM (semiconductor, not a metal) ---
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f'VBM: {vbm:.4f} eV')
print(f'CBM: {cbm:.4f} eV')
print(f'Band gap: {gap:.4f} eV')

# --- Plot ---
fig, axp = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in segments:
    for b in range(E.shape[1]):
        axp.plot(x[s], E[s, b] - vbm, 'k-', lw=0.8)

# Draw a vertical line at every tick
for t in tick_x:
    axp.axvline(t, color='gray', lw=0.5)

# Label the ticks
axp.set_xticks(tick_x)
axp.set_xticklabels(tick_labels)

# Set axis limits
axp.set_xlim(x[0], x[-1])
axp.set_ylim(-10, 10)  # restrict to +/-10 eV around VBM
axp.set_xlabel('k-point')
axp.set_ylabel('Energy (eV)')
axp.set_title('SrTiO3 band structure (PBEsol, aligned to VBM)')

plt.tight_layout()
plt.savefig(OUT_PDF, dpi=150)
print(f'Saved plot to {OUT_PDF}')
