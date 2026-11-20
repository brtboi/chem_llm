#!/usr/bin/env python3
"""Generate 51 GaAs structures: 1 unperturbed base + 50 perturbed copies.

Perturbations: random atomic displacements (thermal motion) + small random
lattice strain, representative of finite-temperature structural noise.

Temperature choice: 300 K (room temperature).
  Zincblende GaAs is the stable phase at ambient conditions (it is the
  equilibrium phase at 300 K; the wurtzite phase is only marginally more
  stable at low T and the rock-salt phase appears only at high pressure).
  300 K is well within the stability range, so thermal displacements at
  this temperature are physically meaningful for this phase.

Displacement amplitudes (per species):
  The crystallographic displacement parameter U is the mean-square
  displacement in A^2. The per-axis RMS displacement is sqrt(U) in A.
  We use the per-axis RMS displacement (sqrt(U)) as the standard deviation
  of a Gaussian distribution for each Cartesian displacement component.

  At 300 K, typical Debye-Waller U values for GaAs (from neutron/XRD data):
    Ga: U ~ 0.008 A^2  -> sqrt(U) ~ 0.089 A  (per-axis RMS)
    As: U ~ 0.006 A^2  -> sqrt(U) ~ 0.077 A  (per-axis RMS)
  Ga is lighter (69.7 amu) than As (74.9 amu), so Ga moves slightly more.
  We use these values directly as the Gaussian sigma for each axis.

Lattice strain: small random strain tensor with components drawn from a
  uniform distribution in [-0.005, 0.005] (0.5% max), representing
  thermal expansion/contraction fluctuations.
"""

import numpy as np
from ase.io import read, write
from ase import Atoms

# Fixed seed for reproducibility
rng = np.random.default_rng(42)

# Read the base structure
base = read('structures/structure_000.cif')
print(f"Base structure: {base.get_chemical_formula()}, {len(base)} atoms")
print(f"Cell: {base.cell[:].tolist()}")

# Per-species RMS displacement (sqrt(U)) in Angstrom at 300 K
# Ga: U ~ 0.008 A^2 -> sqrt(U) ~ 0.089 A
# As: U ~ 0.006 A^2 -> sqrt(U) ~ 0.077 A
DISPLACEMENT_RMS = {
    'Ga': 0.089,  # A, per-axis RMS = sqrt(U), U ~ 0.008 A^2
    'As': 0.077,  # A, per-axis RMS = sqrt(U), U ~ 0.006 A^2
}

# Max strain per component (0.5%)
MAX_STRAIN = 0.005

# Write structure 000: unperturbed base
write('structures/structure_000.cif', base)

# INVARIANT 1 check for base
nat = len(base)
types = set(base.get_chemical_symbols())
ntyp = len(types)
assert nat == 2, f"Expected 2 atoms, got {nat}"
assert ntyp == 2, f"Expected 2 types, got {ntyp}"
print(f"Base: nat={nat}, ntyp={ntyp}")

for i in range(1, 51):
    atoms = base.copy()
    
    # Apply random lattice strain
    # Strain tensor: small random symmetric tensor
    strain = rng.uniform(-MAX_STRAIN, MAX_STRAIN, size=(3, 3))
    # Make it symmetric (strain tensor is symmetric)
    strain = 0.5 * (strain + strain.T)
    # New cell = (I + strain) @ old_cell
    # In ASE, cell is stored as rows, so new_cell = (I + strain) @ cell
    new_cell = (np.eye(3) + strain) @ atoms.cell[:]
    atoms.set_cell(new_cell, scale_atoms=False)
    
    # Apply random atomic displacements (in Cartesian coordinates)
    # Displacements are in Angstrom, applied to fractional positions
    # via the cell matrix
    positions = atoms.get_positions()  # Cartesian, Angstrom
    new_positions = np.zeros_like(positions)
    for idx, (pos, symbol) in enumerate(zip(positions, atoms.get_chemical_symbols())):
        sigma = DISPLACEMENT_RMS[symbol]
        # Gaussian displacement in each Cartesian direction, sigma in A
        disp = rng.normal(0.0, sigma, size=3)
        new_positions[idx] = pos + disp
    atoms.set_positions(new_positions)
    
    # INVARIANT 1 check per structure
    nat_i = len(atoms)
    ntyp_i = len(set(atoms.get_chemical_symbols()))
    assert nat_i == nat, f"Structure {i:03d}: nat {nat_i} != {nat}"
    assert ntyp_i == ntyp, f"Structure {i:03d}: ntyp {ntyp_i} != {ntyp}"
    
    fname = f'structures/structure_{i:03d}.cif'
    write(fname, atoms)

print(f"\nGenerated 51 structures (000-050) in structures/")
print("Done.")
