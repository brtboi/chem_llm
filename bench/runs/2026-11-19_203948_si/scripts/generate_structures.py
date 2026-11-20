#!/usr/bin/env python
"""STEP 2: Generate 51 perturbed Si structures (000 = unperturbed base).

Reads structures/structure_base.cif (MP mp-149, Fd-3m #227, primitive Si2 cell)
and writes structures/structure_000.cif ... structure_050.cif.

Perturbation model: harmonic thermal motion + small random lattice strain.

Temperature choice: 300 K. Diamond-cubic Si is the stable phase of silicon at
ambient conditions (it is the ground-state phase at 300 K and 1 atm; the
metastable beta-Sn phase only becomes stable above ~1000 K / high pressure).
So 300 K is a temperature at which THIS phase is actually stable.

Displacement amplitudes (units made explicit):
  The crystallographic displacement parameter U is a mean-square displacement
  in A^2. The per-axis RMS displacement is sqrt(U) in A.
  We model each Cartesian displacement as a Gaussian with standard deviation
  sigma = sqrt(U) (i.e. sigma is the RMS displacement in A).

  For a harmonic solid at temperature T, equipartition gives the mean-square
  displacement per axis <u^2> = k_B T / k_spring, where k_spring is an
  effective spring constant. We estimate k_spring from the Debye model:
  k_spring ~ M * omega_D^2, with omega_D the Debye frequency.
  For Si: M = 28.0855 g/mol = 4.654e-26 kg, Debye temperature ~645 K.
  omega_D = 2*pi*k_B*T_D/h ~ 8.4e13 rad/s (angular frequency).
  k_spring ~ 4.654e-26 * (8.4e13)^2 ~ 3.2e2 N/m.
  <u^2> = k_B T / k_spring = 1.381e-23 * 300 / 3.2e2 ~ 1.3e-23 m^2 = 1.3e-3 A^2.
  sigma = sqrt(U) = sqrt(1.3e-3) A ~ 0.036 A.

  This is consistent with the experimental Debye-Waller factor for Si at 300 K
  (U ~ 1e-3 A^2). We use sigma = 0.02 A (U ~ 4e-4 A^2) as a slightly milder
  perturbation representative of thermal motion at 300 K, keeping the
  perturbation small enough for a meaningful DFT band-structure comparison.

  Since Si is the only species, all atoms get the same amplitude. If there
  were multiple species, lighter or more weakly bound atoms would get larger
  sigma (sigma ~ 1/sqrt(M) at fixed k_spring, or larger for weaker bonds).

Lattice strain: small random symmetric strain tensor with components of order
  0.1% (1e-3), representative of thermal expansion fluctuations.
"""

import numpy as np
from ase.io import read, write
from ase import Atoms

# Physical constants
k_B = 1.381e-23      # J/K
h = 6.626e-34        # J*s
T = 300.0            # K -- temperature at which diamond-cubic Si is stable
T_debye = 645.0      # K, Debye temperature of Si
M_Si = 28.0855e-3 / 6.022e23  # kg per atom

# Debye angular frequency and effective spring constant
omega_D = 2 * np.pi * k_B * T_debye / h   # rad/s (angular frequency)
k_spring = M_Si * omega_D**2              # N/m

# Mean-square displacement per axis (A^2) and RMS displacement (A)
U = k_B * T / k_spring * 1e20             # m^2 -> A^2 (1 m^2 = 1e20 A^2)
sigma = np.sqrt(U)                        # A, per-axis RMS displacement

# Use a slightly milder value for the perturbation (see docstring)
sigma = 0.02  # A, representative of thermal motion at 300 K for Si

print(f"Temperature: {T} K")
print(f"Debye angular frequency: {omega_D:.3e} rad/s")
print(f"Effective spring constant: {k_spring:.3e} N/m")
print(f"U (mean-square displacement per axis, Debye estimate): {U:.3e} A^2")
print(f"sigma = sqrt(U) (RMS displacement per axis, used): {sigma:.3e} A")

# Strain magnitude: 0.1% random symmetric strain
strain_mag = 1e-3

rng = np.random.default_rng(42)

base = read('structures/structure_base.cif')
print(f"Base structure: {base.get_chemical_formula()}, {len(base)} atoms")

# INVARIANT 1: nat/ntyp computed from the actual Atoms object
nat_base = len(base)
types_base = set(base.get_chemical_symbols())
ntyp_base = len(types_base)
print(f"Base nat={nat_base}, ntyp={ntyp_base}, types={types_base}")

for i in range(51):
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()
        # Random atomic displacements (thermal motion)
        # Displace in Cartesian coordinates, Gaussian with std = sigma (A)
        disp = rng.normal(0.0, sigma, size=(len(atoms), 3))
        atoms.positions += disp
        # Small random lattice strain (symmetric strain tensor)
        strain = rng.normal(0.0, strain_mag, size=(3, 3))
        strain = 0.5 * (strain + strain.T)  # make symmetric
        # Apply strain to the cell: new_cell = (I + strain) @ old_cell
        # In ASE, cell is a 3x3 matrix of row vectors (a1, a2, a3).
        # Strain in Cartesian: new_positions = (I + strain) @ old_positions
        # For the cell: new_cell = (I + strain) @ old_cell (row-vector convention)
        I = np.eye(3)
        atoms.cell = (I + strain) @ atoms.cell
        # Also strain the Cartesian positions (row-vector convention):
        # positions has shape (n_atoms, 3), so new_pos = pos @ (I + strain).T
        atoms.positions = atoms.positions @ (I + strain).T

    # INVARIANT 1: assert nat/ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == nat_base, f"Structure {i:03d}: nat {nat} != base {nat_base}"
    assert ntyp == ntyp_base, f"Structure {i:03d}: ntyp {ntyp} != base {ntyp_base}"

    out_path = f'structures/structure_{i:03d}.cif'
    write(out_path, atoms)
    if i < 3 or i == 50:
        print(f"Wrote {out_path}: nat={nat}, ntyp={ntyp}")

print("Done: wrote 51 structures (000-050).")
