import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# =============================================================================
# plot_bands.py
# Plots the band structure for calculations/000 (fcc Ni, spin-polarized, metal)
# =============================================================================

KPATH_JSON = 'calculations/000/kpath.json'
BANDS_DAT = 'calculations/000/000.bands.dat'
PW_OUT = 'calculations/000/pw.out'
OUTPUT_PDF = 'calculations/000/bands_000.pdf'

# --- Read the band data and x-axis from plot_axis (INVARIANT 4) ---
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']           # path length (arbitrary units)
E = ax['energies']    # shape (n_kpoints, n_bands), in eV
tick_x = ax['tick_x']
tick_labels = ax['tick_labels']
segments = ax['segments']

print(f"Bands: {E.shape[0]} k-points, {E.shape[1]} bands")
print(f"Ticks: {list(zip(tick_x, tick_labels))}")

# --- Read SCF output for electron count and Fermi energy (INVARIANT 2) ---
pw = read_pw_out(PW_OUT)
n_electrons = pw['n_electrons']
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']
spin_orbit = pw['spin_orbit']
print(f"n_electrons = {n_electrons}, n_occupied = {n_occupied}, spin_orbit = {spin_orbit}")
print(f"Fermi energy = {fermi_eV:.4f} eV")

# Ni is a metal (band_gap_eV=0.0 from generate_cif) -> align to Fermi energy
# For a spin-polarized run, n_occupied is the number of occupied bands per spin channel
# (each band holds 1 electron per spin channel)
zero = fermi_eV
print(f"Aligning to Fermi energy: {zero:.4f} eV")

# --- Plot ---
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in segments:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], E[s, b] - zero, 'k-', lw=0.8)

# Draw vertical lines at every tick
for t in tick_x:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Set x-axis ticks and labels
plt_ax.set_xticks(tick_x)
plt_ax.set_xticklabels(tick_labels)

# Set x-limits to the full path
plt_ax.set_xlim(x[0], x[-1])

# Restrict y-axis to about +/-10 eV
plt_ax.set_ylim(-10, 10)

# Labels and title
plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title('fcc Ni band structure (majority spin, aligned to Fermi energy)')

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=150)
print(f"Saved {OUTPUT_PDF}")

# --- Print VBM, CBM, and band gap ---
# For a metal, the "gap" is 0 by definition, but we can still report VBM and CBM
# relative to the Fermi energy
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']
print(f"\nVBM = {vbm:.4f} eV (relative to zero)")
print(f"CBM = {cbm:.4f} eV (relative to zero)")
print(f"Band gap = {gap:.4f} eV")
print(f"Fermi energy = {fermi_eV:.4f} eV")
print(f"\nNote: Ni is a metal, so the band gap is expected to be ~0 eV.")
print(f"The VBM and CBM are reported relative to the Fermi energy (zero).")
