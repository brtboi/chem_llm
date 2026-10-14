#!/usr/bin/env python
"""
Generate 51 perturbed diamond-C structures (structure_000.cif .. structure_050.cif).

structure_000.cif: unperturbed primitive base (already written by reduce_primitive.py).
structure_001.cif .. structure_050.cif: random thermal displacements + small random lattice strain.

Temperature choice: T = 300 K.
  Diamond is the ambient-pressure stable phase of carbon at room temperature.
  It is metastable above ~4000 K where graphite becomes stable, but at 300 K
  diamond is the correct reference phase. 300 K is therefore the natural choice
  for thermal-displacement amplitudes.

Displacement amplitudes (per species):
  Only one species (C) is present.  For diamond at 300 K the mean-square
  displacement U (crystallographic, in A^2) is approximately 0.005 A^2.
  The per-axis RMS displacement is sqrt(U) = sqrt(0.005) ~ 0.0707 A.
  We use sigma = 0.07 A as the standard deviation of the Gaussian displacement
  along each Cartesian axis (i.e. the RMS displacement per axis).

  Justification: diamond's Debye temperature is ~2200 K, so at 300 K the
  vibrational amplitude is small.  A typical U for C in diamond at 300 K is
  ~0.004-0.006 A^2, giving RMS ~0.06-0.08 A.  We use 0.07 A.

Lattice strain:
  Small random strain tensor with components drawn uniformly from [-0.01, 0.01]
  (1 % maximum).  This is well within the elastic stability range of diamond
  (bulk modulus ~440 GPa, so 1 % strain is ~4.4 GPa, far below the ~100 GPa
  fracture strength).

Units:
  - U (crystallographic displacement parameter): A^2 (mean-square displacement)
  - RMS displacement per axis: sqrt(U) in A
  - sigma (Gaussian std dev per axis): A
  - strain: dimensionless (fractional change in lattice parameter)
"""

import numpy as np
import ase.io
from ase import Atoms

# --- Parameters ---
T = 300  # K
# Per-species RMS displacement (A).  Only C is present.
# U = 0.005 A^2  ->  RMS = sqrt(0.005) ~ 0.0707 A
# We use sigma = 0.07 A per axis.
SIGMA_C = 0.07  # A, RMS displacement per Cartesian axis for C at 300 K

# Maximum strain component (dimensionless)
MAX_STRAIN = 0.01  # 1 %

N_STRUCTURES = 51  # 000 (unperturbed) + 001..050 (perturbed)

# --- Read base structure ---
base = ase.io.read('structures/structure_000.cif')
print(f'Base structure: {len(base)} atoms, cell = {base.cell[:]}')

# INVARIANT 1: nat and ntyp computed from the actual Atoms object
nat = len(base)
types = set(base.get_chemical_symbols())
ntyp = len(types)
print(f'nat = {nat}, ntyp = {ntyp}, species = {types}')
assert nat == 2, f'Expected 2 atoms in primitive diamond cell, got {nat}'
assert ntyp == 1, f'Expected 1 species, got {ntyp}'

# --- Generate structures ---
np.random.seed(42)  # reproducibility

for i in range(N_STRUCTURES):
    if i == 0:
        # Unperturbed base
        atoms = base.copy()
    else:
        atoms = base.copy()

        # 1. Random lattice strain
        #    Strain tensor: symmetric 3x3, components in [-MAX_STRAIN, MAX_STRAIN]
        strain = np.random.uniform(-MAX_STRAIN, MAX_STRAIN, size=(3, 3))
        # Make it symmetric (physical strain is symmetric)
        strain = 0.5 * (strain + strain.T)
        # New cell = (I + strain) @ old_cell
        #    In ASE, cell is a 3x3 matrix of row vectors (a, b, c).
        #    The strain acts as: new_cell = (I + strain) @ old_cell
        #    (strain in Cartesian, cell vectors as rows)
        new_cell = (np.eye(3) + strain) @ np.array(atoms.cell)
        atoms.cell = new_cell

        # 2. Random atomic displacements (thermal motion)
        #    Gaussian, per-axis sigma = SIGMA_C A for C
        #    Displacements are in Cartesian coordinates (A)
        displacements = np.random.normal(0.0, SIGMA_C, size=atoms.positions.shape)
        atoms.positions += displacements

    # INVARIANT 1: verify nat/ntyp for each structure
    assert len(atoms) == nat, f'Structure {i:03d}: nat mismatch'
    assert len(set(atoms.get_chemical_symbols())) == ntyp, f'Structure {i:03d}: ntyp mismatch'

    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)
    print(f'Wrote {out_path} ({len(atoms)} atoms)')

print(f'\nDone. Generated {N_STRUCTURES} structures.')
