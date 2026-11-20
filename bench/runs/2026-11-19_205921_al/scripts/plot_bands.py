#!/usr/bin/env python
"""
STEP 7: Plot the band structure for calculations/000 (fcc Al, mp-134).

Al is a metal (band_gap=0.0 per Materials Project), so we align the bands
to the Fermi energy read from pw.out via chem_llm.dft.qe_output.read_pw_out.

Per INVARIANT 4: x-axis, ticks, and line segments come from
chem_llm.dft.kpath.plot_axis, which also verifies that bands.x computed
the same k-point list as kpath.json.

Per INVARIANT 2: n_occupied comes from read_pw_out (ground truth from the
SCF run), not from a guessed valence-electron count.
"""

import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.dft.kpath import plot_axis
from chem_llm.dft.qe_output import read_pw_out, band_edges

CALC_DIR = 'calculations/000'
KPATH_JSON = f'{CALC_DIR}/kpath.json'
BANDS_DAT = f'{CALC_DIR}/000.bands.dat'
PW_OUT = f'{CALC_DIR}/pw.out'
OUT_PDF = f'{CALC_DIR}/bands_000.pdf'


def main():
    # --- Read the SCF output for Fermi energy and occupied-band count ---
    pw = read_pw_out(PW_OUT)
    n_occupied = pw['n_occupied']
    fermi_eV = pw['fermi_energy_eV']
    print(f"n_occupied = {n_occupied}")
    print(f"Fermi energy = {fermi_eV:.4f} eV")

    # Al is a metal: align to the Fermi energy
    zero = fermi_eV
    print(f"Aligning to Fermi energy: {zero:.4f} eV")

    # --- Get x-axis, energies, ticks, segments from plot_axis (INVARIANT 4) ---
    ax = plot_axis(KPATH_JSON, BANDS_DAT)
    x = ax['x']           # path length array
    E = ax['energies']    # (n_kpts, n_bands) in eV
    segments = ax['segments']   # list of slice objects, one per continuous segment
    tick_x = ax['tick_x']      # list of tick positions
    tick_labels = ax['tick_labels']  # list of tick labels

    print(f"Path length: {x[-1]:.3f} A^-1")
    print(f"Number of bands: {E.shape[1]}")
    print(f"Number of k-points: {E.shape[0]}")
    print(f"Segments: {len(segments)}")
    print(f"Ticks: {list(zip(tick_x, tick_labels))}")

    # --- Plot ---
    fig, axp = plt.subplots(figsize=(10, 6))

    # Plot each band as one line per segment (never across a discontinuity)
    for s in segments:
        for b in range(E.shape[1]):
            axp.plot(x[s], E[s, b] - zero, 'k-', lw=1)

    # Draw a vertical line at every tick
    for t in tick_x:
        axp.axvline(t, color='gray', lw=0.5)

    # Label the ticks
    axp.set_xticks(tick_x)
    axp.set_xticklabels(tick_labels)

    # Set x-limits to the full path
    axp.set_xlim(x[0], x[-1])

    # Restrict y-axis to about +/-10 eV around the Fermi level
    axp.set_ylim(-10, 10)

    # Labels and title
    axp.set_xlabel('k-path (A$^{-1}$)')
    axp.set_ylabel('Energy - E$_F$ (eV)')
    axp.set_title('Al (mp-134) band structure -- PBE PAW, QE, aligned to Fermi level')

    plt.tight_layout()
    plt.savefig(OUT_PDF, dpi=150)
    print(f"Saved plot to {OUT_PDF}")

    # --- Print VBM, CBM, and band gap ---
    # For a metal, the gap is 0 by definition, but we still report the
    # VBM and CBM relative to the Fermi level for reference.
    edges = band_edges(E, n_occupied)
    vbm = edges['vbm'] - zero
    cbm = edges['cbm'] - zero
    gap = edges['gap']
    print(f"\nVBM (relative to E_F): {vbm:.4f} eV")
    print(f"CBM (relative to E_F): {cbm:.4f} eV")
    print(f"Band gap: {gap:.4f} eV")
    print(f"Fermi energy (absolute): {fermi_eV:.4f} eV")


if __name__ == '__main__':
    main()
