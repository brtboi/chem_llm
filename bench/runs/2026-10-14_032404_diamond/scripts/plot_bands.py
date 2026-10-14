import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# =============================================================================
# plot_bands.py
# Plots the band structure for calculations/000 (diamond C, mp-66).
# Aligns to VBM (diamond is an insulator, gap ~4.1 eV from MP).
# =============================================================================

KPATH_JSON = 'calculations/000/kpath.json'
BANDS_DAT = 'calculations/000/000.bands.dat'
PW_OUT = 'calculations/000/pw.out'
OUTPUT_PDF = 'calculations/000/bands_000.pdf'

# --- Get x-axis, energies, ticks, segments from plot_axis (INVARIANT 4) ---
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']
E = ax['energies']  # shape (n_kpts, n_bands), energies in eV
print(f"Band data: {E.shape[0]} k-points, {E.shape[1]} bands")
print(f"Path length: {x[0]:.4f} to {x[-1]:.4f}")
print(f"Ticks: {ax['tick_labels']}")

# --- Read pw.out for electron count and Fermi energy (INVARIANT 2) ---
pw = read_pw_out(PW_OUT)
n_electrons = pw['n_electrons']
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']
print(f"Number of electrons: {n_electrons}")
print(f"Number of occupied bands: {n_occupied}")
if fermi_eV is not None:
    print(f"Fermi energy: {fermi_eV:.4f} eV")
else:
    print("Fermi energy: not reported (insulator with fixed occupations)")

# INVARIANT 2: check nbnd against occupied bands
# For spin-unpolarized non-metallic run: occupied bands = n_electrons / 2
expected_occupied = int(n_electrons / 2)
assert n_occupied == expected_occupied, \
    f"n_occupied={n_occupied} != n_electrons/2={expected_occupied}"
print(f"INVARIANT 2 check: n_occupied={n_occupied} == n_electrons/2={expected_occupied} ✓")

# --- Align to VBM (diamond is an insulator) ---
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f"VBM: {vbm:.4f} eV")
print(f"CBM: {cbm:.4f} eV")
print(f"Band gap: {gap:.4f} eV")

# --- Plot ---
fig, ax_plot = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (INVARIANT 4: never across a discontinuity)
for s in ax['segments']:
    for b in range(E.shape[1]):
        ax_plot.plot(x[s], E[s, b] - vbm, 'k-', lw=1)

# Draw vertical lines at every tick (INVARIANT 4)
for t in ax['tick_x']:
    ax_plot.axvline(t, color='gray', lw=0.5)

# Label ticks from plot_axis (INVARIANT 4)
ax_plot.set_xticks(ax['tick_x'])
ax_plot.set_xticklabels(ax['tick_labels'])

# Set x-limits to full path
ax_plot.set_xlim(x[0], x[-1])

# Restrict y-axis to about +/-10 eV (INVARIANT 4)
ax_plot.set_ylim(-10, 10)

# Labels and title
ax_plot.set_xlabel('k-path')
ax_plot.set_ylabel('Energy (eV) relative to VBM')
ax_plot.set_title('Diamond C (mp-66) - Band Structure (PBEsol, scalar relativistic)')

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=150)
print(f"Saved plot to {OUTPUT_PDF}")

# --- Print summary ---
print("\n=== Summary ===")
print(f"Materials Project ID: mp-66")
print(f"Space group: Fd-3m (#227)")
print(f"Primitive cell: 2 atoms")
print(f"VBM: {vbm:.4f} eV")
print(f"CBM: {cbm:.4f} eV")
print(f"Band gap: {gap:.4f} eV")
if fermi_eV is not None:
    print(f"Fermi energy: {fermi_eV:.4f} eV")
else:
    print("Fermi energy: not reported (insulator with fixed occupations)")
print(f"Number of electrons: {n_electrons}")
print(f"Number of occupied bands: {n_occupied}")
