#!/usr/bin/env python3
"""
Generate 51 perturbed diamond-Si structures (structure_000.cif .. structure_050.cif).

structure_000.cif: unperturbed primitive cell (2 atoms, Fd-3m #227, mp-149).
structure_001..050: random atomic displacements (thermal motion) + small random lattice strain.

Temperature choice: 300 K.
  Diamond Si is the thermodynamically stable phase from 0 K up to the melting point
  (~1687 K at 1 atm), so 300 K is well within its stability range.

Displacement model: Einstein oscillator, one amplitude per species.
  Per-axis RMS displacement: sigma = sqrt(k_B * T / (M * omega_E^2))
  where M is the atomic mass in kg and omega_E is the Einstein frequency.
  For Si, the Debye temperature is ~645 K, so omega_E ~ k_B * Theta_D / hbar ~ 8.4e13 rad/s.
  sigma = sqrt(1.380649e-23 * 300 / (4.663e-26 * (8.4e13)^2)) ~ 0.02 A
  Total mean-square displacement (crystallographic U): U = 3 * sigma^2 ~ 0.0012 A^2
  Per-axis RMS = sqrt(U/3) = sigma ~ 0.02 A

  Only one species (Si), so one amplitude. Lighter or more weakly bound atoms
  would get a larger amplitude; here all atoms are Si.

Strain: small random symmetric strain tensor, magnitude ~0.5% (diagonal and off-diagonal
  components drawn from uniform distribution in [-0.005, 0.005]).
"""

import numpy as np
from ase.io import read, write
from chem_llm.kpath import standard_kpath, kp_atoms

# --- Constants ---
k_B = 1.380649e-23  # J/K
hbar = 1.054571817e-34  # J*s
T = 300.0           # K
M_Si = 28.0855e-3 / 6.02214076e23  # kg per atom
Theta_D_Si = 645.0  # K (Debye temperature of diamond Si)

# Einstein frequency from Debye temperature
omega_E = k_B * Theta_D_Si / hbar  # rad/s

# Per-axis RMS displacement (Einstein model): sigma = sqrt(k_B*T / (M*omega_E^2))
sigma_Si = np.sqrt(k_B * T / (M_Si * omega_E**2))  # in meters
sigma_Si_A = sigma_Si * 1e10  # convert to Angstrom
U_Si_A2 = 3.0 * sigma_Si_A**2  # total MSD in A^2 (crystallographic U)

print(f"Temperature: {T} K")
print(f"Einstein frequency omega_E: {omega_E:.4e} rad/s")
print(f"Si per-axis RMS displacement sigma: {sigma_Si_A:.6f} A")
print(f"Si total MSD U = 3*sigma^2: {U_Si_A2:.6e} A^2")

# Sanity check: sigma should be in the range 0.01-0.05 A for Si at 300 K
assert 0.005 < sigma_Si_A < 0.05, f"sigma_Si_A = {sigma_Si_A} A is outside expected range"

# --- Read base structure and reduce to primitive cell ---
base = read('structures/base_Si.cif')
kp = standard_kpath(base)
prim = kp_atoms(kp)

print(f"Primitive cell: {len(prim)} atoms, space group {kp['spacegroup']}")
assert len(prim) == 2, "Diamond Si primitive cell must have 2 atoms"

# --- Write structure_000.cif (unperturbed) ---
write('structures/structure_000.cif', prim)
print("Wrote structures/structure_000.cif (unperturbed)")

# --- Generate perturbed structures 001..050 ---
np.random.seed(42)  # reproducibility

for i in range(1, 51):
    atoms = prim.copy()

    # Random atomic displacements (Cartesian, in Angstrom)
    # Each atom gets a 3D Gaussian displacement with per-axis std = sigma_Si_A
    displacements = np.random.normal(0.0, sigma_Si_A, size=(len(atoms), 3))
    atoms.positions += displacements

    # Small random lattice strain (symmetric tensor, ~0.5%)
    strain = np.random.uniform(-0.005, 0.005, size=(3, 3))
    strain = 0.5 * (strain + strain.T)  # symmetrize
    # Apply strain: new_cell = (I + strain) @ old_cell
    # In ASE, cell rows are the lattice vectors; strain acts on the basis
    new_cell = (np.eye(3) + strain) @ atoms.cell[:]
    atoms.cell = new_cell

    # INVARIANT 1: assert nat/ntyp computed from actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Structure {i:03d}: expected 2 atoms, got {nat}"
    assert ntyp == 1, f"Structure {i:03d}: expected 1 species, got {ntyp}"

    fname = f'structures/structure_{i:03d}.cif'
    write(fname, atoms)

print("Wrote structures/structure_001.cif through structure_050.cif")
print("Done.")
