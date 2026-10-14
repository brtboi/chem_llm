import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# =============================================================================
# plot_bands.py
# Plot the band structure for calculations/000 (fcc Ni, metal).
# Ni is a metal (band_gap_eV = 0.0 from generate_cif), so we align to the
# Fermi energy read from pw.out (INVARIANT 2, RULE: occupations).
# =============================================================================

KPATH_JSON = 'calculations/000/kpath.json'
BANDS_DAT = 'calculations/000/000.bands.dat'
PW_OUT = 'calculations/000/pw.out'
OUT_PDF = 'calculations/000/bands_000.pdf'

# --- Read the band energies and x-axis from plot_axis (INVARIANT 4) ---
# plot_axis checks that bands.x computed the same k-point list as kpath.json.
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']          # path length (A^-1) for each k-point
E = ax['energies']   # shape (n_kpts, n_bands), in eV
tick_x = ax['tick_x']
tick_labels = ax['tick_labels']
segments = ax['segments']  # slice objects, one per line segment

# --- Read SCF output for Fermi energy and electron count (INVARIANT 2) ---
pw = read_pw_out(PW_OUT)
n_electrons = pw['n_electrons']
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']

print(f"Number of electrons: {n_electrons}")
print(f"Number of occupied bands: {n_occupied}")
print(f"Fermi energy: {fermi_eV:.4f} eV")

# Ni is a metal -> align to Fermi energy
zero = fermi_eV

# --- Plot ---
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in segments:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], E[s, b] - zero, 'k-', lw=1)

# Draw a vertical line at every tick
for t in tick_x:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Label the ticks from plot_axis (they already read 'A|L' at discontinuities)
plt_ax.set_xticks(tick_x)
plt_ax.set_xticklabels(tick_labels)

# Set x-limits to the full path
plt_ax.set_xlim(x[0], x[-1])

# Restrict y-axis to about +/-10 eV (INVARIANT 4)
plt_ax.set_ylim(-10, 10)

plt_ax.set_xlabel('k-point path (A$^{-1}$)')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title('Band structure of fcc Ni (PBEsol, scalar-relativistic)\n' +
                 f'Fermi energy = {fermi_eV:.4f} eV')

plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches='tight')
print(f"Saved band structure plot to {OUT_PDF}")

# --- Print VBM, CBM, and band gap (or Fermi energy for a metal) ---
# For a metal, the "gap" is 0 by definition; we report the Fermi energy.
# band_edges gives VBM and CBM relative to the zero we chose.
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']

print(f"\n--- Band edges (relative to Fermi energy) ---")
print(f"VBM: {vbm - zero:.4f} eV")
print(f"CBM: {cbm - zero:.4f} eV")
print(f"Band gap: {gap:.4f} eV")
print(f"Fermi energy (absolute): {fermi_eV:.4f} eV")
print("\nNi is a metal: the Fermi level lies within the band manifold.")
