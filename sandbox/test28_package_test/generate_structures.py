#!/usr/bin/env python3
"""
Generate 51 perturbed TiO2 structures from the primitive cell.

Structure 000: unperturbed base structure.
Structures 001-050: random atomic displacements + small random lattice strain.

Displacement magnitude justification:
  At room temperature (300 K), the mean-square displacement of atoms in rutile TiO2
  is on the order of 0.01-0.05 Angstrom (from Debye-Waller factors in neutron/XRD data).
  We use a Gaussian displacement with sigma = 0.05 A, which is at the upper end of
  typical thermal vibrations and ensures the perturbed structures remain chemically
  reasonable (no bond breaking, no unphysical overlaps).

Lattice strain justification:
  A small random strain of up to 1% (sigma = 0.005) is applied to the cell vectors.
  This is well within the elastic range for rutile TiO2 and simulates the kind of
  small lattice distortions that can arise from thermal expansion or defects.
"""

import numpy as np
from ase.io import read, write
from ase import Atoms

# Read the primitive cell base structure
base = read('structures/base_TiO2_primitive.cif')
print(f"Base structure: {len(base)} atoms, cell = {base.cell}")
print(f"Elements: {base.get_chemical_symbols()}")

# Displacement parameters
DISPLACEMENT_SIGMA = 0.05  # Angstrom, Gaussian sigma for atomic displacements
STRAIN_SIGMA = 0.005       # 0.5% Gaussian sigma for lattice strain (up to ~1%)

# Random seed for reproducibility
np.random.seed(42)

for i in range(51):
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()
        
        # Apply random atomic displacements (Gaussian, sigma = 0.05 A)
        # Displacements are in Cartesian coordinates
        displacements = np.random.normal(0, DISPLACEMENT_SIGMA, size=atoms.positions.shape)
        atoms.positions += displacements
        
        # Apply small random lattice strain
        # Strain tensor: symmetric, small, applied to the cell vectors
        # cell = cell @ (I + strain)^T  (strain is a 3x3 matrix)
        strain = np.random.normal(0, STRAIN_SIGMA, size=(3, 3))
        # Make strain symmetric (physical strain tensor is symmetric)
        strain = 0.5 * (strain + strain.T)
        # Apply strain to the cell: new_cell = old_cell @ (I + strain)^T
        # In ASE, cell vectors are rows, so cell @ (I + strain).T
        new_cell = atoms.cell @ (np.eye(3) + strain).T
        atoms.cell = new_cell
    
    # INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    
    # Assert nat/ntyp are consistent with the base structure
    assert nat == len(base), f"Structure {i:03d}: nat changed from {len(base)} to {nat}"
    assert ntyp == len(set(base.get_chemical_symbols())), \
        f"Structure {i:03d}: ntyp changed from {len(set(base.get_chemical_symbols()))} to {ntyp}"
    
    # Write the structure
    out_path = f'structures/structure_{i:03d}.cif'
    write(out_path, atoms)
    
    if i == 0 or i % 10 == 0:
        print(f"Wrote {out_path}: {nat} atoms, {ntyp} types")

print(f"\nDone. Generated 51 structures (000-050).")
print(f"Base: {len(base)} atoms, {len(set(base.get_chemical_symbols()))} types")
