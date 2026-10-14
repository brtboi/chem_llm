import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# =============================================================================
# plot_bands.py
# Plots the band structure for calculations/000 (bcc Fe, spin-polarized)
# =============================================================================

# --- Read the band data and k-path axis ---
kpath_json = 'calculations/000/kpath.json'
bands_dat = 'calculations/000/000.bands.dat'

ax = plot_axis(kpath_json, bands_dat)
x = ax['x']
E = ax['energies']  # shape: (nkpts, nbands)
print(f"Band data: {E.shape[0]} k-points, {E.shape[1]} bands")
print(f"Path length: {x[0]:.3f} to {x[-1]:.3f}")

# --- Read SCF output for Fermi energy and occupied bands ---
pw_out = 'calculations/000/pw.out'
pw = read_pw_out(pw_out)
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']
print(f"n_occupied: {n_occupied}")
print(f"Fermi energy: {fermi_eV:.4f} eV")
print(f"Spin-orbit: {pw['spin_orbit']}")
print(f"Total magnetization: {pw['total_magnetization']}")

# --- Align energies ---
# Fe is a metal (FM), so align to the Fermi energy
zero = fermi_eV
print(f"Aligning to Fermi energy: {zero:.4f} eV")

# --- Compute band edges for reference ---
edges = band_edges(E, n_occupied)
print(f"VBM: {edges['vbm']:.4f} eV")
print(f"CBM: {edges['cbm']:.4f} eV")
print(f"Band gap: {edges['gap']:.4f} eV")

# --- Plot ---
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in ax['segments']:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], E[s, b] - zero, 'k-', lw=1)

# Draw vertical lines at every tick
for t in ax['tick_x']:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Set tick labels
plt_ax.set_xticks(ax['tick_x'])
plt_ax.set_xticklabels(ax['tick_labels'])

# Set axis limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)

# Labels and title
plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title('bcc Fe band structure (majority spin)\nAligned to Fermi energy')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print("Saved plot to calculations/000/bands_000.pdf")

# --- Print summary ---
print("\n=== Summary ===")
print(f"Fermi energy: {fermi_eV:.4f} eV")
print(f"VBM: {edges['vbm']:.4f} eV")
print(f"CBM: {edges['cbm']:.4f} eV")
print(f"Band gap: {edges['gap']:.4f} eV")
print(f"Total magnetization: {pw['total_magnetization']}")
