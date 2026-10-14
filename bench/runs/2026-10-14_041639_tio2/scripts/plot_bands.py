import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# =============================================================================
# plot_bands.py
# Plots the band structure for calculations/000 (rutile TiO2, primitive cell)
# =============================================================================

# Read the band data and k-path axis (INVARIANT 4)
ax = plot_axis('calculations/000/kpath.json', 'calculations/000/000.bands.dat')
x = ax['x']          # path length (A^-1)
E = ax['energies']   # (nkpts, nbands) in eV

# Read SCF output for electron count and Fermi energy (INVARIANT 2)
pw = read_pw_out('calculations/000/pw.out')
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']

print(f"Number of electrons: {pw['n_electrons']}")
print(f"Number of occupied bands: {n_occupied}")
if fermi_eV is not None:
    print(f"Fermi energy: {fermi_eV:.4f} eV")
else:
    print("Fermi energy: not available (fixed occupations, insulator)")

# Align to VBM (insulator, gap 1.77 eV from MP)
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']

print(f"VBM: {vbm:.4f} eV")
print(f"CBM: {cbm:.4f} eV")
print(f"Band gap: {gap:.4f} eV")

# Plot bands: one line per segment (never across a discontinuity)
fig, plt_ax = plt.subplots(figsize=(10, 6))

for s in ax['segments']:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], E[s, b] - vbm, 'k-', lw=1)

# Draw vertical lines at every tick
for t in ax['tick_x']:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Label ticks from plot_axis (they already read 'A|L' at discontinuities)
plt_ax.set_xticks(ax['tick_x'])
plt_ax.set_xticklabels(ax['tick_labels'])

# Set x-limits to full path
plt_ax.set_xlim(x[0], x[-1])

# Restrict y-axis to about +/-10 eV
plt_ax.set_ylim(-10, 10)

plt_ax.set_xlabel('k-point path (A$^{-1}$)')
plt_ax.set_ylabel('Energy relative to VBM (eV)')
plt_ax.set_title('Rutile TiO$_2$ band structure (PBEsol, scalar relativistic)')
plt_ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print("Saved calculations/000/bands_000.pdf")
