#!/usr/bin/env python3
"""
Generate 51 perturbed fcc Cu structures (structure_000.cif ... structure_050.cif).

Structure 000 is the unperturbed primitive cell.
Structures 001-050 have random atomic displacements (thermal motion) and a small random lattice strain.

Temperature choice:
  fcc Cu is stable at room temperature (T_melt = 1358 K). We use T = 300 K, well below melting,
  so the fcc phase is the stable equilibrium phase.

Displacement amplitudes:
  For a monatomic fcc metal, the mean-square displacement (U) can be estimated from the Debye model.
  For Cu at 300 K, the Debye temperature is ~343 K, so kT << hbar*omega_D and the displacements are small.
  A typical RMS displacement for Cu at 300 K is ~0.01-0.02 A.
  We use a per-axis RMS displacement of 0.015 A (i.e., U = (0.015 A)^2 = 2.25e-4 A^2 per axis).
  This is representative of thermal motion at 300 K for a heavy metal like Cu.

  Since Cu is monatomic, all atoms have the same amplitude.

Strain:
  A small random homogeneous strain tensor (max ~0.5%) is applied to each perturbed structure.
  This simulates small lattice distortions that might occur in a real crystal.

Units:
  - Displacements: Angstrom (A)
  - RMS displacement per axis: sqrt(U) in A, where U is the mean-square displacement in A^2
  - Strain: dimensionless
"""

import numpy as np
import ase.io
from ase import Atoms
from ase.build import fcc111  # not used, just for reference

# Read the primitive cell
atoms = ase.io.read('structures/primitive.cif')
print(f"Base structure: {len(atoms)} atoms, cell = {atoms.get_cell()}")

# Write structure_000 (unperturbed)
ase.io.write('structures/structure_000.cif', atoms)
print("Wrote structures/structure_000.cif (unperturbed)")

# Parameters for perturbation
T = 300.0  # K, temperature at which fcc Cu is stable (T_melt = 1358 K)
# RMS displacement per axis in Angstrom
# For Cu at 300 K, typical RMS displacement is ~0.01-0.02 A
# We use 0.015 A per axis (U = (0.015)^2 = 2.25e-4 A^2 per axis)
rms_disp = 0.015  # A, per-axis RMS displacement = sqrt(U)
U = rms_disp ** 2  # A^2, mean-square displacement per axis

# Strain: max strain component ~0.5%
max_strain = 0.005

rng = np.random.default_rng(seed=42)

for i in range(1, 51):
    # Start from a fresh copy of the base structure
    perturbed = Atoms(
        symbols=atoms.get_chemical_symbols(),
        positions=atoms.get_positions().copy(),
        cell=atoms.get_cell().copy(),
        pbc=True
    )

    # Apply random atomic displacements (thermal motion)
    # Each atom gets a random displacement drawn from a Gaussian with std = rms_disp
    # This is equivalent to U = (rms_disp)^2 per axis
    displacements = rng.normal(loc=0.0, scale=rms_disp, size=perturbed.get_positions().shape)
    perturbed.positions += displacements

    # Apply small random homogeneous strain
    # Strain tensor: symmetric, traceless (volume-preserving) for simplicity
    # Generate a random symmetric matrix and subtract its trace/3 to make it traceless
    strain = rng.uniform(-max_strain, max_strain, size=(3, 3))
    strain = (strain + strain.T) / 2.0  # make symmetric
    strain -= np.trace(strain) / 3.0 * np.eye(3)  # make traceless (volume-preserving)

    # Apply strain to the cell
    # new_cell = (I + strain) @ old_cell
    strain_matrix = np.eye(3) + strain
    new_cell = strain_matrix @ perturbed.get_cell()
    perturbed.set_cell(new_cell, scale_atoms=True)

    # INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
    nat = len(perturbed)
    ntyp = len(set(perturbed.get_chemical_symbols()))
    assert nat == 1, f"Expected 1 atom, got {nat}"
    assert ntyp == 1, f"Expected 1 type, got {ntyp}"

    # Write the perturbed structure
    fname = f'structures/structure_{i:03d}.cif'
    ase.io.write(fname, perturbed)
    print(f"Wrote {fname}: nat={nat}, ntyp={ntyp}, max_disp={np.max(np.abs(displacements)):.4f} A")

print("\nDone. Generated 51 structures (000-050).")
