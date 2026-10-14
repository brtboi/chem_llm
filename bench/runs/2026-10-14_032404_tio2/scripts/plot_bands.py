import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# =============================================================================
# plot_bands.py
# Plots the band structure for calculations/000 (rutile TiO2, PBEsol, SCF+NSCF)
# =============================================================================

# Read the k-path axis and energies from the bands.dat file
# plot_axis checks that the bands were computed on the k-points in kpath.json
ax = plot_axis('calculations/000/kpath.json', 'calculations/000/000.bands.dat')
x = ax['x']          # path length (A^-1) for each k-point
E = ax['energies']   # shape (n_kpts, n_bands), in eV

# Read SCF output for electron count and Fermi energy
pw = read_pw_out('calculations/000/pw.out')
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']

print(f"Number of electrons: {pw['n_electrons']}")
print(f"Number of occupied bands: {n_occupied}")
# Fermi energy is None when occupations='fixed' (no smearing), which is
# expected for a clear semiconductor. We align to the VBM instead.
if fermi_eV is not None:
    print(f"Fermi energy: {fermi_eV:.4f} eV")
else:
    print("Fermi energy: not reported (occupations='fixed', no smearing)")

# Rutile TiO2 is a clear semiconductor (gap ~3 eV experimental, ~1.77 eV DFT-PBE)
# Align to VBM (not Fermi energy) per the occupations rule
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']

print(f"VBM: {vbm:.4f} eV")
print(f"CBM: {cbm:.4f} eV")
print(f"Band gap: {gap:.4f} eV")

# Align energies to VBM (set VBM = 0)
E_aligned = E - vbm

# Plot
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in ax['segments']:
    for b in range(E_aligned.shape[1]):
        plt_ax.plot(x[s], E_aligned[s, b], 'k-', lw=1)

# Draw vertical lines at every tick
for t in ax['tick_x']:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Set x-axis ticks and labels from plot_axis
plt_ax.set_xticks(ax['tick_x'])
plt_ax.set_xticklabels(ax['tick_labels'])

# Set axis limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)  # restrict to +/-10 eV around VBM

# Labels and title
plt_ax.set_xlabel('k-path (A$^{-1}$)')
plt_ax.set_ylabel('Energy (eV)')
plt_ax.set_title('Rutile TiO$_2$ band structure (PBEsol, SCF+NSCF)\n' +
                 f'VBM = 0 eV, CBM = {cbm - vbm:.2f} eV, gap = {gap:.2f} eV')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print("Saved calculations/000/bands_000.pdf")
