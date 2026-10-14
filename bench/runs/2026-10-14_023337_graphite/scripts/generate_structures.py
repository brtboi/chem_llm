#!/usr/bin/env python3
"""
Generate 51 perturbed graphite structures (structure_000.cif .. structure_050.cif).

Structure 000 is the unperturbed primitive cell.
Structures 001-050 have:
  - random atomic displacements (Einstein model, thermal motion)
  - small random lattice strain

Temperature choice: T = 1000 K
  Graphite is the thermodynamically stable phase of carbon at ambient pressure
  up to the sublimation point (~3900 K). At 1000 K it is well within its
  stability range. The Debye temperature of graphite is ~1500 K, so at 1000 K
  the atoms are in the classical (high-T) regime where the Einstein model
  gives a reasonable estimate of mean-square displacements.

Einstein model for mean-square displacement:
  <u^2> = 3 * hbar^2 / (m * omega_E^2) * [1/2 + 1/(exp(hbar*omega_E/kT) - 1)]
  For T >> Debye temperature, this approaches the classical limit:
  <u^2> = 3 * kT / (m * omega_E^2)

We use the classical limit (valid at T=1000 K for graphite, Debye ~1500 K):
  <u^2> = 3 * kT / (m * omega_E^2)

For graphite, the in-plane C-C stretching frequency is ~1560 cm^-1 (omega_E ~ 3.0e13 rad/s).
  The out-of-plane (flexural) mode is much softer, ~100 cm^-1 (omega_E ~ 2.0e12 rad/s).
  We use an effective omega_E that gives a reasonable RMS displacement.

  Carbon mass: m_C = 12.011 u = 1.993e-26 kg
  k_B = 1.381e-23 J/K
  T = 1000 K

  For in-plane: omega_E = 2*pi*1560*100 = 9.8e13 rad/s (converting cm^-1 to s^-1: *2*pi*c)
    Actually: omega = 2*pi*nu, nu = c * wavenumber = 3e10 cm/s * 1560 cm^-1 = 4.68e13 Hz
    omega = 2*pi*4.68e13 = 2.94e14 rad/s
    <u^2> = 3 * 1.381e-23 * 1000 / (1.993e-26 * (2.94e14)^2)
          = 4.143e-20 / (1.993e-26 * 8.64e28)
          = 4.143e-20 / 1.722e3
          = 2.41e-23 m^2 = 2.41e-15 A^2
    RMS per axis = sqrt(2.41e-15/3) = sqrt(8.03e-16) = 2.83e-8 A  -- too small!

  Let me recalculate more carefully.
  nu = c * wavenumber = 2.998e10 cm/s * 1560 cm^-1 = 4.677e13 Hz
  omega = 2*pi*nu = 2.938e14 rad/s
  m*omega^2 = 1.993e-26 * (2.938e14)^2 = 1.993e-26 * 8.632e28 = 1720 kg/s^2
  3*kT = 3 * 1.381e-23 * 1000 = 4.143e-20 J
  <u^2> = 4.143e-20 / 1720 = 2.41e-23 m^2 = 2.41e-15 A^2
  Per-axis RMS = sqrt(<u^2>/3) = sqrt(8.03e-16) A^2 = 2.83e-8 A

  This is indeed very small because the in-plane mode is very stiff.
  For the out-of-plane (flexural) mode at ~100 cm^-1:
  nu = 2.998e10 * 100 = 2.998e12 Hz
  omega = 2*pi*2.998e12 = 1.884e13 rad/s
  m*omega^2 = 1.993e-26 * (1.884e13)^2 = 1.993e-26 * 3.55e26 = 7.07 kg/s^2
  <u^2> = 4.143e-20 / 7.07 = 5.86e-21 m^2 = 5.86e-13 A^2
  Per-axis RMS = sqrt(5.86e-13/3) = sqrt(1.95e-13) = 4.42e-7 A

  These are still very small. In practice, for DFT perturbation studies,
  we use larger displacements to represent the range of thermal motion.
  A typical approach is to use RMS displacements of ~0.01-0.05 A for
  thermal motion at moderate temperatures.

  For this workflow, we use:
    - In-plane (x,y) RMS displacement: 0.02 A  (representative of thermal motion at ~1000 K)
    - Out-of-plane (z) RMS displacement: 0.05 A  (flexural modes are softer, larger displacement)

  These are per-axis RMS values in Angstroms.
  The corresponding mean-square displacement U (in A^2) would be:
    U_inplane = 0.02^2 = 4e-4 A^2
    U_outofplane = 0.05^2 = 2.5e-3 A^2

Lattice strain: small random strain of magnitude ~0.5% (0.005) in each component,
  representing thermal expansion/contraction and small structural distortions.
"""

import numpy as np
import ase.io
from ase import Atoms
from ase.spacegroup import crystal
import os

# Random seed for reproducibility
np.random.seed(42)

# Read the base primitive cell
base = ase.io.read('structures/base_primitive.cif')
print(f"Base structure: {len(base)} atoms, symbols: {base.get_chemical_symbols()}")

# Temperature and displacement parameters (see docstring for justification)
T = 1000  # K
# Per-axis RMS displacements in Angstroms
rms_inplane = 0.02   # A, for x and y directions (stiff in-plane modes)
rms_outofplane = 0.05  # A, for z direction (softer flexural modes)

# Lattice strain magnitude (dimensionless, ~0.5%)
strain_mag = 0.005

# Output directory
out_dir = 'structures'
os.makedirs(out_dir, exist_ok=True)

# Structure 000: unperturbed base
ase.io.write(os.path.join(out_dir, 'structure_000.cif'), base)
print("Wrote structure_000.cif (unperturbed)")

# Structures 001-050: perturbed
for i in range(1, 51):
    atoms = base.copy()
    
    # Apply random lattice strain
    # Strain tensor: small random symmetric matrix
    strain = np.random.normal(0, strain_mag, size=(3, 3))
    # Make it symmetric (strain tensor is symmetric)
    strain = 0.5 * (strain + strain.T)
    # Apply strain to cell: new_cell = (I + strain) @ old_cell
    # In ASE, cell vectors are rows, so we need to be careful
    # new_cell_vectors = (I + strain.T) @ old_cell_vectors
    old_cell = atoms.cell.array  # shape (3,3), rows are cell vectors
    new_cell = (np.eye(3) + strain.T) @ old_cell
    atoms.cell = new_cell
    
    # Apply random atomic displacements
    # Displacements are in Cartesian coordinates (Angstroms)
    # Each atom gets a random displacement from a Gaussian distribution
    # with the appropriate RMS for its direction
    positions = atoms.get_positions().copy()
    for j in range(len(atoms)):
        # In-plane displacement (x, y)
        dx = np.random.normal(0, rms_inplane)
        dy = np.random.normal(0, rms_inplane)
        # Out-of-plane displacement (z)
        dz = np.random.normal(0, rms_outofplane)
        positions[j] += np.array([dx, dy, dz])
    atoms.set_positions(positions)
    
    # INVARIANT 1: assert nat and ntyp are consistent
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 4, f"Structure {i:03d}: expected 4 atoms, got {nat}"
    assert ntyp == 1, f"Structure {i:03d}: expected 1 element type, got {ntyp}"
    
    # Write the structure
    fname = os.path.join(out_dir, f'structure_{i:03d}.cif')
    ase.io.write(fname, atoms)
    
    if i <= 3 or i == 50:
        print(f"Wrote {fname}")

print(f"\nDone. Generated 51 structures in {out_dir}/")
print("Structure 000: unperturbed base")
print("Structures 001-050: perturbed with thermal displacements + random strain")
print(f"  T = {T} K")
print(f"  RMS in-plane displacement: {rms_inplane} A")
print(f"  RMS out-of-plane displacement: {rms_outofplane} A")
print(f"  Strain magnitude: {strain_mag} ({strain_mag*100:.1f}%)")
