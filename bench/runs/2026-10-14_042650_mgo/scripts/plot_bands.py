import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chem_llm.kpath import plot_axis
from chem_llm.qe_output import read_pw_out, band_edges

# ============================================================
# plot_bands.py
# Plots the band structure for calculations/000 (MgO, unperturbed).
# Energies aligned to VBM (insulator, no smearing).
# ============================================================

kpath_json = 'calculations/000/kpath.json'
bands_dat = 'calculations/000/000.bands.dat'
pw_out = 'calculations/000/pw.out'

# --- Get x-axis, energies, ticks, segments from plot_axis (INVARIANT 4) ---
ax = plot_axis(kpath_json, bands_dat)
x = ax['x']          # path length, shape (N_kpts,)
E = ax['energies']   # shape (N_kpts, N_bands), in eV
segments = ax['segments']   # list of slice objects, one per continuous segment
tick_x = ax['tick_x']       # list of float x positions
tick_labels = ax['tick_labels']  # list of strings

# --- Read SCF output for electron count and Fermi energy (INVARIANT 2) ---
pw = read_pw_out(pw_out)
n_occupied = pw['n_occupied']  # number of occupied bands (spin-unpolarized: n_electrons/2)
fermi_eV = pw['fermi_energy_eV']

print(f"n_electrons = {pw['n_electrons']}")
print(f"n_occupied  = {n_occupied}")
if fermi_eV is not None:
    print(f"fermi_energy = {fermi_eV:.4f} eV")
else:
    print("fermi_energy = None (insulator with fixed occupations, no smearing)")

# --- Align to VBM (insulator) ---
edges = band_edges(E, n_occupied)
vbm = edges['vbm']
cbm = edges['cbm']
gap = edges['gap']

print(f"VBM = {vbm:.4f} eV")
print(f"CBM = {cbm:.4f} eV")
print(f"Band gap = {gap:.4f} eV")

# --- Plot ---
fig, axp = plt.subplots(figsize=(8, 5))

for s in segments:
    for b in range(E.shape[1]):
        axp.plot(x[s], E[s, b] - vbm, 'k-', lw=1)

for t in tick_x:
    axp.axvline(t, color='gray', lw=0.5)

axp.set_xticks(tick_x)
axp.set_xticklabels(tick_labels)
axp.set_xlim(x[0], x[-1])
axp.set_ylim(-10, 10)
axp.set_xlabel('k-point')
axp.set_ylabel('Energy (eV) relative to VBM')
axp.set_title('MgO band structure (PBEsol, scalar-relativistic)')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf')
print("Saved calculations/000/bands_000.pdf")
