#!/usr/bin/env python3
"""Plot the band structure for calculations/000 (GaAs, mp-2534).

Reads:
  - calculations/000/kpath.json (MP's k-path)
  - calculations/000/000.bands.dat (from bands.x)
  - calculations/000/pw.out (SCF output, for n_occupied and Fermi energy)

Writes:
  - calculations/000/bands_000.pdf

Prints VBM, CBM, and band gap (eV).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.dft.kpath import plot_axis
from chem_llm.dft.qe_output import read_pw_out, band_edges

KPATH_JSON = 'calculations/000/kpath.json'
BANDS_DAT = 'calculations/000/000.bands.dat'
PW_OUT = 'calculations/000/pw.out'
OUTPUT_PDF = 'calculations/000/bands_000.pdf'

# Get x-axis, energies, ticks, and line segments from plot_axis
# This also verifies that bands.dat was computed on the k-points in kpath.json
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']          # path length (float array)
E = ax['energies']   # shape (n_kpts, n_bands), in eV
tick_x = ax['tick_x']
tick_labels = ax['tick_labels']
segments = ax['segments']  # list of slice objects, one per continuous segment

# Read SCF output for n_occupied and Fermi energy
pw = read_pw_out(PW_OUT)
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']

print(f"n_occupied = {n_occupied}")
# For occupations='fixed' (semiconductor), QE does not compute a Fermi energy;
# fermi_energy_eV is None. Only print it if available.
if fermi_eV is not None:
    print(f"Fermi energy = {fermi_eV:.4f} eV")
else:
    print("Fermi energy = None (occupations='fixed', no smearing)")

# Align to VBM (semiconductor, occupations='fixed')
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']

print(f"VBM = {vbm:.4f} eV")
print(f"CBM = {cbm:.4f} eV")
print(f"Band gap = {gap:.4f} eV")

# Plot
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

# Set limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)

plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title('GaAs (mp-2534) band structure, PBE PAW, no spin-orbit')

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=150)
print(f"Saved plot to {OUTPUT_PDF}")
