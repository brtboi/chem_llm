import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# =============================================================================
# plot_bands.py
# Plots the band structure for calculations/000 (fcc Ni, spin-polarized).
# Ni is a metal -> align to the Fermi energy.
# =============================================================================

KPATH_JSON = 'calculations/000/kpath.json'
BANDS_DAT = 'calculations/000/000.bands.dat'
PW_OUT = 'calculations/000/pw.out'
OUT_PDF = 'calculations/000/bands_000.pdf'

# --- Read QE output (INVARIANT 2: use chem_llm.qe_output, not hand-written regex) ---
pw = read_pw_out(PW_OUT)
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']
print(f"n_occupied = {n_occupied}")
print(f"Fermi energy = {fermi_eV:.4f} eV")
print(f"Total magnetization = {pw.get('total_magnetization', 'N/A')} muB")

# --- Get x-axis, energies, ticks, segments from plot_axis (INVARIANT 4) ---
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']
E = ax['energies']  # shape: (n_kpts, n_bands)
print(f"x shape: {x.shape}, E shape: {E.shape}")

# Ni is a metal -> align to Fermi energy
zero = fermi_eV

# --- Plot ---
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in ax['segments']:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], E[s, b] - zero, 'k-', lw=1)

# Draw vertical line at every tick
for t in ax['tick_x']:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Label ticks from plot_axis (they already read 'A|L' at discontinuities)
plt_ax.set_xticks(ax['tick_x'])
plt_ax.set_xticklabels(ax['tick_labels'])

# Set x-limits to the full path
plt_ax.set_xlim(x[0], x[-1])

# Restrict y-axis to about +/-10 eV
plt_ax.set_ylim(-10, 10)

plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy - E$_F$ (eV)')
plt_ax.set_title('Ni band structure (majority spin, spin-polarized)')

plt.tight_layout()
plt.savefig(OUT_PDF, dpi=150)
print(f"Saved plot to {OUT_PDF}")

# --- Print band edges ---
# For a metal, the gap is 0 by definition, but let's report the Fermi energy
# and the VBM/CBM relative to it for reference.
edges = band_edges(E, n_occupied)
print(f"\nBand edges (relative to VBM):")
print(f"  VBM = {edges['vbm']:.4f} eV")
print(f"  CBM = {edges['cbm']:.4f} eV")
print(f"  Gap = {edges['gap']:.4f} eV")
print(f"\nFermi energy (alignment reference) = {fermi_eV:.4f} eV")
print(f"VBM relative to E_F = {edges['vbm'] - fermi_eV:.4f} eV")
print(f"CBM relative to E_F = {edges['cbm'] - fermi_eV:.4f} eV")
