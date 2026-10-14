import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# =============================================================================
# plot_bands.py
# Plot the band structure for calculations/000 (2H MoS2, unperturbed base).
# Uses chem_llm.kpath.plot_axis for the x-axis and chem_llm.qe_output for
# reading the SCF output (n_occupied, fermi_energy_eV) per INVARIANT 2.
# =============================================================================

KPATH_JSON = 'calculations/000/kpath.json'
BANDS_DAT = 'calculations/000/000.bands.dat'
PW_OUT = 'calculations/000/pw.out'
OUTPUT_PDF = 'calculations/000/bands_000.pdf'

# Get x-axis, energies, ticks, and line segments from plot_axis
# This checks that bands were computed on kpath.json's k-points (INVARIANT 4)
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']          # path length array
E = ax['energies']   # (nkpts, nbands) array in eV
tick_x = ax['tick_x']
tick_labels = ax['tick_labels']
segments = ax['segments']  # list of slice objects

# Read SCF output to get n_occupied and fermi_energy_eV (INVARIANT 2)
pw = read_pw_out(PW_OUT)
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']

# Determine alignment: VBM for insulator/semiconductor, Fermi for metal
# MoS2 2H is a semiconductor (gap ~1.46 eV from MP), so align to VBM
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']

print(f"n_occupied = {n_occupied}")
if fermi_eV is not None:
    print(f"Fermi energy = {fermi_eV:.4f} eV")
else:
    print("Fermi energy = None (not reported for fixed occupations)")
print(f"VBM = {vbm:.4f} eV")
print(f"CBM = {cbm:.4f} eV")
print(f"Band gap = {gap:.4f} eV")

# Align energies to VBM (semiconductor)
zero = vbm

# Create the plot
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in segments:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], E[s, b] - zero, 'k-', lw=1)

# Draw vertical lines at every tick
for t in tick_x:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Set x-axis ticks and labels
plt_ax.set_xticks(tick_x)
plt_ax.set_xticklabels(tick_labels)

# Set axis limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)  # restrict to +/-10 eV around VBM

# Labels and title
plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title('2H MoS$_2$ band structure (PBEsol, scalar-relativistic)\n' +
                 f'Gap = {gap:.3f} eV (VBM-aligned)')

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=150, bbox_inches='tight')
print(f"Saved band structure plot to {OUTPUT_PDF}")
