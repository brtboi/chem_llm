import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# =============================================================================
# plot_bands.py
# Plots the band structure for calculations/000 (PbTe, spin-orbit, noncolin).
# Energies are aligned to the VBM (PbTe is a semiconductor, gap ~0.8 eV).
# With spin-orbit, each band holds 1 electron, so n_occupied = n_electrons.
# =============================================================================

KPATH_JSON = 'calculations/000/kpath.json'
BANDS_DAT = 'calculations/000/000.bands.dat'
PW_OUT = 'calculations/000/pw.out'
OUTPUT_PDF = 'calculations/000/bands_000.pdf'

# Get x-axis, energies, ticks, and line segments from plot_axis
# INVARIANT 4: plot_axis checks that bands were computed on kpath.json's k-points
ax = plot_axis(KPATH_JSON, BANDS_DAT)
x = ax['x']
E = ax['energies']  # shape: (n_kpts, n_bands), energies in eV

print(f"x-axis: {len(x)} points, range [{x[0]:.4f}, {x[-1]:.4f}]")
print(f"Energies: {E.shape[0]} k-points, {E.shape[1]} bands")
print(f"Ticks: {ax['tick_labels']}")
print(f"Segments: {ax['segments']}")

# Read pw.out to get n_occupied and fermi_energy_eV
# INVARIANT 2: use chem_llm.qe_output, not hand-written regex
pw = read_pw_out(PW_OUT)
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']
n_electrons = pw['n_electrons']
spin_orbit = pw['spin_orbit']

print(f"\npw.out info:")
print(f"  n_electrons = {n_electrons}")
print(f"  n_occupied = {n_occupied}")
# fermi_energy_eV can be None for non-smearing runs (occupations='fixed')
if fermi_eV is not None:
    print(f"  fermi_energy_eV = {fermi_eV:.4f}")
else:
    print(f"  fermi_energy_eV = None (not reported for fixed occupations)")
print(f"  spin_orbit = {spin_orbit}")

# INVARIANT 2 check: with spin-orbit, n_occupied should equal n_electrons
# (each band holds 1 electron, not 2)
if spin_orbit:
    assert n_occupied == n_electrons, \
        f"With spin-orbit, n_occupied ({n_occupied}) should equal n_electrons ({n_electrons})"
    print(f"  INVARIANT 2 check passed: n_occupied == n_electrons (spin-orbit)")
else:
    assert n_occupied == n_electrons / 2, \
        f"Without spin-orbit, n_occupied ({n_occupied}) should equal n_electrons/2 ({n_electrons/2})"
    print(f"  INVARIANT 2 check passed: n_occupied == n_electrons/2 (no spin-orbit)")

# Align energies to the VBM (PbTe is a semiconductor)
# band_edges returns vbm, cbm, gap in eV
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']

print(f"\nBand edges (before alignment):")
print(f"  VBM = {vbm:.4f} eV")
print(f"  CBM = {cbm:.4f} eV")
print(f"  Gap = {gap:.4f} eV")

# Shift energies so VBM is at 0
eV_shifted = E - vbm

# Plot
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
# INVARIANT 4: segments are slice objects
for s in ax['segments']:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], eV_shifted[s, b], 'k-', lw=1)

# Draw vertical lines at every tick
for t in ax['tick_x']:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Label the ticks
plt_ax.set_xticks(ax['tick_x'])
plt_ax.set_xticklabels(ax['tick_labels'])

# Set x-limits to the full path
plt_ax.set_xlim(x[0], x[-1])

# Restrict y-axis to about +/-10 eV
plt_ax.set_ylim(-10, 10)

# Labels and title
if fermi_eV is not None:
    title = (f'PbTe band structure (spin-orbit, VBM at 0 eV)\n'
             f'Gap = {gap:.3f} eV, Fermi = {fermi_eV:.3f} eV')
else:
    title = (f'PbTe band structure (spin-orbit, VBM at 0 eV)\n'
             f'Gap = {gap:.3f} eV')
plt_ax.set_title(title)

plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV)')

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=150)
print(f"\nSaved plot to {OUTPUT_PDF}")

print(f"\nSummary:")
print(f"  VBM = {vbm:.4f} eV")
print(f"  CBM = {cbm:.4f} eV")
print(f"  Band gap = {gap:.4f} eV")
if fermi_eV is not None:
    print(f"  Fermi energy = {fermi_eV:.4f} eV")
else:
    print(f"  Fermi energy = None (not reported for fixed occupations)")
