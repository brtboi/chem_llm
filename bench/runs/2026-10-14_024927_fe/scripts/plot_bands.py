#!/usr/bin/env python
"""
STEP 7: Plot the band structure for calculations/000 (bcc Fe, FM, spin-polarized).

- x-axis, energies, ticks, and line segments come from chem_llm.kpath.plot_axis
  (INVARIANT 4: it checks that bands.x computed the same k-points as kpath.json).
- Energies are aligned to the Fermi energy (Fe is a metal, FM -> smearing in SCF).
  n_occupied and fermi_energy_eV are read from pw.out via chem_llm.qe_output
  (INVARIANT 2: never assume the occupied-band count from a textbook valence).
- Each band is drawn as one line per segment (no line across a discontinuity).
- Vertical line + tick label at every tick from plot_axis.
- y-axis restricted to about +/-10 eV around the Fermi energy.
- Saves calculations/000/bands_000.pdf and prints VBM, CBM, band gap, Fermi energy.
"""

import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

CALC_DIR = 'calculations/000'
KPATH_JSON = f'{CALC_DIR}/kpath.json'
BANDS_DAT = f'{CALC_DIR}/000.bands.dat'
PW_OUT = f'{CALC_DIR}/pw.out'
OUT_PDF = f'{CALC_DIR}/bands_000.pdf'

# --- Read the band data and the SCF output ---
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']          # path length (arbitrary units, one per k-point)
E = ax['energies']   # shape (nkpts, nbands), energies in eV

pw = read_pw_out(PW_OUT)
n_occupied = pw['n_occupied']   # occupied-band count from the actual pw.out
fermi_eV = pw['fermi_energy_eV']  # Fermi energy from the smearing SCF

# Fe is a metal (FM, smearing in SCF) -> align to the Fermi energy.
zero = fermi_eV

# Band edges relative to the Fermi energy
edges = band_edges(E, n_occupied)
vbm = edges['vbm'] - zero   # VBM relative to Fermi
# CBM: for a metal the Fermi level lies inside the bands; report the lowest
# band energy above the Fermi energy as a reference (not a true gap).
cbm = edges['cbm'] - zero   # CBM relative to Fermi (may be negative for a metal)
gap = edges['gap']          # band gap in eV (0 for a metal)

print(f"Fermi energy: {fermi_eV:.4f} eV")
print(f"VBM (relative to Fermi): {vbm:.4f} eV")
print(f"CBM (relative to Fermi): {cbm:.4f} eV")
print(f"Band gap: {gap:.4f} eV")
print(f"n_occupied (from pw.out): {n_occupied}")

# --- Plot ---
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Draw each band as one line per segment (INVARIANT 4: never across a discontinuity)
for s in ax['segments']:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], E[s, b] - zero, 'k-', lw=1)

# Vertical line at every tick
for t in ax['tick_x']:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Tick labels from plot_axis (already read 'A|L' at discontinuities)
plt_ax.set_xticks(ax['tick_x'])
plt_ax.set_xticklabels(ax['tick_labels'])

# Axis limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)  # about +/-10 eV around the Fermi energy

# Labels and title
plt_ax.set_xlabel('k-path')
plt_ax.set_ylabel('Energy - E$_F$ (eV)')
plt_ax.set_title('bcc Fe (FM, majority spin) - band structure\n' +
                 f'Fermi energy = {fermi_eV:.3f} eV, gap = {gap:.3f} eV')

plt.tight_layout()
plt.savefig(OUT_PDF, dpi=150)
print(f"Saved plot to {OUT_PDF}")
