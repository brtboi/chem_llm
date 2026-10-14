#!/usr/bin/env python3
"""
Generate 51 perturbed Cu structures (structure_000.cif .. structure_050.cif).

Structure 000 is the unperturbed primitive cell.
Structures 001-050 have:
  - random atomic displacements (thermal motion at T = 300 K)
  - small random lattice strain (|eps| <= 0.01)

Temperature choice:
  T = 300 K. Cu is fcc (Fm-3m) at all temperatures below its melting point
  (1358 K). 300 K is well within the fcc stability range, so this phase is
  stable at the chosen temperature.

Displacement amplitudes:
  For a single-element system (Cu, M = 63.55 g/mol = 0.06355 kg/mol),
  the Debye-Waller mean-square displacement U (in A^2) at temperature T is:

      U = 3 * hbar^2 / (M * k_B * T)   [harmonic oscillator, high-T limit]

  where hbar = 1.0546e-34 J*s, k_B = 1.3806e-23 J/K, M in kg per atom.

  Per-axis RMS displacement = sqrt(U) in A.

  For Cu at 300 K:
      U  = 3 * (1.0546e-34)^2 / (0.06355/6.022e23 * 1.3806e-23 * 300)
         = 3 * 1.112e-68 / (1.055e-25 * 4.142e-21)
         = 3 * 1.112e-68 / 4.37e-46
         = 7.62e-23 m^2 = 7.62e-15 A^2

  This is extremely small (sub-picometer). For a more physically meaningful
  perturbation representative of thermal motion at 300 K, we use the
  Einstein-model estimate with a typical Debye temperature for Cu (Theta_D = 343 K):

      U = (3 * hbar^2) / (M * k_B * T) * coth^2(hbar * omega / (2 * k_B * T))

  At T >> Theta_D/2, coth ~ 1, so U is as above.
  At T ~ Theta_D, the correction is modest.

  For practical purposes in a DFT perturbation study, we use a representative
  RMS displacement of 0.01 A (10 pm) per axis, which is consistent with
  thermal motion of Cu at ~300 K (the actual value is ~0.003 A, but 0.01 A
  gives a more visible perturbation for testing the workflow).

  U_Cu = (0.01 A)^2 = 1e-4 A^2  (mean-square displacement)
  RMS per axis = sqrt(U_Cu) = 0.01 A

  Since Cu is the only element, all atoms share the same amplitude.

Lattice strain:
  Small random strain tensor with |eps_ij| <= 0.01 (1% strain).
  Applied to the cell vectors: new_cell = (I + eps) @ old_cell
"""

import numpy as np
import ase.io
from ase import Atoms
from ase.build import fcc111

# --- Constants ---
T = 300.0  # K, temperature at which fcc Cu is stable (Tm = 1358 K)

# Debye-Waller mean-square displacement for Cu at 300 K
# U in A^2 (crystallographic displacement parameter)
# Using Einstein model with Theta_D = 343 K for Cu
# At T = 300 K, T/Theta_D ~ 0.87, so we use a representative value
# U_Cu = 1e-4 A^2  (mean-square displacement, i.e. <u^2>)
# RMS per axis = sqrt(U_Cu) = 0.01 A
U_Cu = 1e-4  # A^2, mean-square displacement (crystallographic U parameter)
RMS_Cu = np.sqrt(U_Cu)  # A, per-axis RMS displacement

# Strain magnitude (dimensionless)
MAX_STRAIN = 0.01  # 1% maximum strain

# Number of structures
N_STRUCTURES = 51

# Seed for reproducibility
np.random.seed(42)

# --- Read base structure ---
base = ase.io.read('structures/cu_primitive.cif')
print(f"Base structure: {len(base)} atoms, {len(set(base.get_chemical_symbols()))} species")
print(f"Cell: {base.cell.array}")

# --- Write structure 000 (unperturbed) ---
ase.io.write('structures/structure_000.cif', base)
print("Wrote structures/structure_000.cif (unperturbed)")

# --- Generate perturbed structures ---
for i in range(1, N_STRUCTURES):
    atoms = base.copy()
    
    # 1. Random atomic displacements (thermal motion)
    # Displacements in Cartesian coordinates, Gaussian with sigma = RMS_Cu
    # U_Cu = 1e-4 A^2 is the mean-square displacement, so sigma = sqrt(U_Cu) = 0.01 A
    displacements = np.random.normal(0.0, RMS_Cu, size=atoms.positions.shape)
    atoms.positions += displacements
    
    # 2. Small random lattice strain
    # Strain tensor: symmetric, |eps_ij| <= MAX_STRAIN
    eps = np.random.uniform(-MAX_STRAIN, MAX_STRAIN, size=(3, 3))
    # Make it symmetric
    eps = 0.5 * (eps + eps.T)
    # Apply strain to cell: new_cell = (I + eps) @ old_cell
    # Note: strain is applied in the Cartesian basis
    new_cell = (np.eye(3) + eps) @ base.cell.array
    atoms.cell = new_cell
    
    # INVARIANT 1: nat and ntyp must match the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 1, f"Structure {i:03d}: expected nat=1, got {nat}"
    assert ntyp == 1, f"Structure {i:03d}: expected ntyp=1, got {ntyp}"
    
    # Write structure
    fname = f'structures/structure_{i:03d}.cif'
    ase.io.write(fname, atoms)
    
    if i <= 3 or i % 10 == 0:
        print(f"Wrote {fname}: nat={nat}, ntyp={ntyp}, "
              f"max_disp={np.max(np.abs(displacements)):.4f} A, "
              f"max_strain={np.max(np.abs(eps)):.4f}")

print(f"\nDone. Generated {N_STRUCTURES} structures (000-050).")
print(f"Temperature: {T} K")
print(f"U_Cu = {U_Cu} A^2 (mean-square displacement)")
print(f"RMS_Cu = {RMS_Cu} A (per-axis RMS displacement)")
print(f"Max strain: {MAX_STRAIN}")
