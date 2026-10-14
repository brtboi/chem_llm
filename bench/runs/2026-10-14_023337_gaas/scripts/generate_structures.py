#!/usr/bin/env python3
"""
Generate 51 GaAs structures: structure_000.cif (unperturbed) + 50 perturbed copies.

Perturbation model:
  - Random atomic displacements representative of thermal motion at 300 K.
  - Small random lattice strain (~0.5% per component).

Temperature choice: 300 K
  Zincblende GaAs is the stable phase at room temperature. The phase transition
  to wurtzite occurs only under high pressure (>~8 GPa) or at very high
  temperatures (>~1500 K). 300 K is well within the stability range.

Displacement amplitudes (Einstein model):
  RMS displacement per axis = sqrt(U) where U = kBT / (m * omega^2)
  For GaAs at 300 K, typical Debye temperature ~300 K gives RMS ~ 0.05-0.08 A.
  Lighter Ga (69.7 amu) moves slightly more than As (74.9 amu).
  Per-species RMS displacement (in Angstroms):
    Ga: 0.060 A  (sqrt(U) = 0.060 A, so U = 0.0036 A^2)
    As: 0.055 A  (sqrt(U) = 0.055 A, so U = 0.003025 A^2)

Strain: random symmetric strain tensor with components drawn from N(0, 0.005)
  (0.5% strain per component, representative of small thermal expansion fluctuations).

Units: all displacements in Angstroms, strain dimensionless.
"""

import numpy as np
import ase.io
from ase import Atoms

# Read the base (unperturbed) structure
base = ase.io.read('structures/structure_000.cif')
print(f"Base structure: {len(base)} atoms, symbols: {base.get_chemical_symbols()}")

# Per-species RMS displacement in Angstroms (sqrt(U), where U is in A^2)
# Ga is lighter (69.7 amu) than As (74.9 amu), so Ga moves slightly more.
RMS_GA = 0.060  # A  (U_Ga = 0.0036 A^2)
RMS_AS = 0.055  # A  (U_As = 0.003025 A^2)

# Strain standard deviation (dimensionless, ~0.5%)
STRAIN_STD = 0.005

# Number of perturbed structures (000 is unperturbed, so 50 perturbed)
N_PERTURBED = 50

# Fixed seed for reproducibility
rng = np.random.default_rng(42)

# Write structure_000 (already done by reduce_cell.py, but ensure it exists)
ase.io.write('structures/structure_000.cif', base)
print("Wrote structures/structure_000.cif (unperturbed)")

for i in range(1, N_PERTURBED + 1):
    atoms = base.copy()
    
    # --- Random atomic displacements (thermal motion) ---
    # Displacements are in Cartesian coordinates (Angstroms).
    # Each atom gets an independent Gaussian displacement with per-species RMS.
    symbols = atoms.get_chemical_symbols()
    displacements = np.zeros((len(atoms), 3))
    for j, sym in enumerate(symbols):
        rms = RMS_GA if sym == 'Ga' else RMS_AS
        displacements[j] = rng.normal(0.0, rms, size=3)
    
    # Apply displacements to Cartesian positions
    positions = atoms.get_positions() + displacements
    atoms.set_positions(positions)
    
    # --- Small random lattice strain ---
    # Generate a random symmetric strain tensor: epsilon = (S + S^T) / 2
    # where S is a 3x3 matrix with components ~ N(0, STRAIN_STD)
    S = rng.normal(0.0, STRAIN_STD, size=(3, 3))
    epsilon = (S + S.T) / 2.0
    
    # Apply strain to the cell: new_cell = (I + epsilon) @ old_cell
    # In ASE, cell vectors are rows, so: new_cell = old_cell @ (I + epsilon)^T
    # But strain is conventionally applied as: new_cell = (I + epsilon) @ old_cell
    # Since cell vectors are rows in ASE, we need: new_cell = old_cell @ (I + epsilon)^T
    I = np.eye(3)
    new_cell = atoms.cell.array @ (I + epsilon).T
    atoms.set_cell(new_cell, scale_atoms=True)  # scale_atoms=True keeps fractional coords
    
    # --- INVARIANT 1: assert nat and ntyp are consistent ---
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Expected 2 atoms, got {nat}"
    assert ntyp == 2, f"Expected 2 types, got {ntyp}"
    
    # Write the perturbed structure
    label = f"{i:03d}"
    out_path = f"structures/structure_{label}.cif"
    ase.io.write(out_path, atoms)
    
    if i <= 3 or i % 10 == 0:
        print(f"Wrote {out_path} (nat={nat}, ntyp={ntyp})")

print(f"\nDone. Generated {N_PERTURBED + 1} structures total (000-050).")
