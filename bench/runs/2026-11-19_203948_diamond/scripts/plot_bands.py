#!/usr/bin/env python3
"""STEP 7: Plot the band structure for calculations/000 (diamond C, mp-66).

Uses:
  - chem_llm.dft.kpath.plot_axis for x-axis, ticks, segments (INVARIANT 4)
  - chem_llm.dft.qe_output.read_pw_out for n_occupied, fermi_energy_eV (INVARIANT 2)
  - chem_llm.dft.qe_output.band_edges for VBM/CBM/gap
  - matplotlib for the plot

Aligns to VBM (diamond is a clear insulator, gap ~4.1 eV per MP).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.dft.kpath import plot_axis
from chem_llm.dft.qe_output import read_pw_out, band_edges

CALC_DIR = 'calculations/000'
KPATH_JSON = f'{CALC_DIR}/kpath.json'
BANDS_DAT = f'{CALC_DIR}/000.bands.dat'
PW_OUT = f'{CALC_DIR}/pw.out'
OUTPUT_PDF = f'{CALC_DIR}/bands_000.pdf'


def main():
    # --- Read pw.out for electron count and Fermi energy (INVARIANT 2) ---
    pw = read_pw_out(PW_OUT)
    n_occupied = pw['n_occupied']
    fermi_eV = pw['fermi_energy_eV']  # None for fixed occupations (no smearing)
    n_electrons = pw['n_electrons']

    # Print fermi energy only if it is available (None for fixed occupations)
    fermi_str = f'{fermi_eV:.4f} eV' if fermi_eV is not None else 'N/A (fixed occupations)'
    print(f'From pw.out: n_electrons={n_electrons}, n_occupied={n_occupied}, '
          f'fermi_energy={fermi_str}')

    # INVARIANT 2 check: for nspin=1, n_occupied should be n_electrons / 2
    assert abs(n_occupied - n_electrons / 2) < 0.5, \
        f'n_occupied={n_occupied} != n_electrons/2={n_electrons/2}'

    # --- Get x-axis, energies, ticks, segments from plot_axis (INVARIANT 4) ---
    ax = plot_axis(KPATH_JSON, BANDS_DAT)
    x = ax['x']           # path length array, shape (n_kpts,)
    E = ax['energies']    # energies in eV, shape (n_kpts, n_bands)
    segments = ax['segments']  # list of slice objects
    tick_x = ax['tick_x']     # list of x positions for ticks
    tick_labels = ax['tick_labels']  # list of tick labels

    print(f'plot_axis: {len(x)} k-points, {E.shape[1]} bands, '
          f'{len(segments)} segments, {len(tick_x)} ticks')

    # --- Align to VBM (insulator) ---
    edges = band_edges(E, n_occupied)
    vbm = edges['vbm']
    cbm = edges['cbm']
    gap = edges['gap']
    zero = vbm  # align VBM to 0 eV

    print(f'VBM = {vbm:.4f} eV, CBM = {cbm:.4f} eV, gap = {gap:.4f} eV')

    # --- Plot ---
    fig, ax_plot = plt.subplots(figsize=(10, 6))

    # Plot each band as one line per segment (never across a discontinuity)
    for s in segments:
        for b in range(E.shape[1]):
            ax_plot.plot(x[s], E[s, b] - zero, 'k-', lw=1)

    # Draw vertical lines at every tick
    for t in tick_x:
        ax_plot.axvline(t, color='gray', lw=0.5)

    # Label ticks
    ax_plot.set_xticks(tick_x)
    ax_plot.set_xticklabels(tick_labels)

    # Set limits
    ax_plot.set_xlim(x[0], x[-1])
    ax_plot.set_ylim(-10, 10)  # ~+/-10 eV around VBM

    ax_plot.set_xlabel('k-point')
    ax_plot.set_ylabel('Energy (eV)')
    ax_plot.set_title('Diamond C (mp-66) -- PBE PAW, no spin-orbit\n'
                      f'VBM=0, gap={gap:.3f} eV (MP: 4.115 eV)')

    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, bbox_inches='tight')
    print(f'Plot saved to {OUTPUT_PDF}')


if __name__ == '__main__':
    main()
