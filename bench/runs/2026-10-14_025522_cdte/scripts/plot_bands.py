import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# Read the band structure data and k-path axis
ax = plot_axis('calculations/000/kpath.json', 'calculations/000/000.bands.dat')
x = ax['x']
E = ax['energies']  # shape (nkpts, nbands), energies in eV

# Read SCF output for electron count and Fermi energy
pw = read_pw_out('calculations/000/pw.out')
n_occupied = pw['n_occupied']
fermi_eV = pw['fermi_energy_eV']

print(f"Number of electrons: {pw['n_electrons']}")
print(f"Number of occupied bands: {n_occupied}")

# CdTe is a semiconductor (band_gap_eV=0.5867 from MP), so align to VBM.
# For a fixed-occupation run, fermi_energy_eV may be None -- that's fine,
# we use the VBM as the zero of energy.
if fermi_eV is not None:
    print(f"Fermi energy: {fermi_eV:.4f} eV")
else:
    print("Fermi energy: not reported (fixed occupations, insulator)")

edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']

print(f"VBM: {vbm:.4f} eV")
print(f"CBM: {cbm:.4f} eV")
print(f"Band gap: {gap:.4f} eV")

# Plot: align energies to VBM (zero at VBM)
fig, plt_ax = plt.subplots(figsize=(10, 6))

# Plot each band as one line per segment (never across a discontinuity)
for s in ax['segments']:
    for b in range(E.shape[1]):
        plt_ax.plot(x[s], E[s, b] - vbm, 'k-', lw=1)

# Draw vertical lines at every tick
for t in ax['tick_x']:
    plt_ax.axvline(t, color='gray', lw=0.5)

# Label ticks from plot_axis
plt_ax.set_xticks(ax['tick_x'])
plt_ax.set_xticklabels(ax['tick_labels'])

# Set limits
plt_ax.set_xlim(x[0], x[-1])
plt_ax.set_ylim(-10, 10)
plt_ax.set_xlabel('k-point')
plt_ax.set_ylabel('Energy (eV) relative to VBM')
plt_ax.set_title('CdTe band structure (PBEsol, scalar-relativistic)')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf')
print("Saved calculations/000/bands_000.pdf")
