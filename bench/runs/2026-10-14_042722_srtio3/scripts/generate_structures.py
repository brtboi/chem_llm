#!/usr/bin/env python
"""
Generate 51 perturbed SrTiO3 structures (structure_000.cif through structure_050.cif).

Structure 000: unperturbed base (already written by reduce_cell.py).
Structures 001-050: random thermal displacements + small random lattice strain.

Temperature choice: T = 300 K
  SrTiO3 cubic (Pm-3m) phase is stable above the ferroelectric transition
  temperature Tc ~ 105 K. At 300 K the cubic phase is the stable phase.

Thermal displacement model:
  Each atom gets a random displacement drawn from a Gaussian distribution
  with per-axis RMS amplitude sigma_i = sqrt(U_i), where U_i is the
  crystallographic mean-square displacement parameter (in A^2) for species i.

  Typical U values at 300 K for cubic SrTiO3 (from neutron diffraction):
    O:  U ~ 0.015 A^2  -> sigma = sqrt(0.015) ~ 0.122 A
    Ti: U ~ 0.008 A^2  -> sigma = sqrt(0.008) ~ 0.089 A
    Sr: U ~ 0.004 A^2  -> sigma = sqrt(0.004) ~ 0.063 A

  Lighter atoms (O, 16 amu) move more than heavier atoms (Sr, 88 amu),
  consistent with equipartition: <U> ~ 3kBT/(m omega^2), and O has a lower
  force constant in the perovskite lattice.

Lattice strain:
  Small random symmetric strain tensor (|epsilon_ij| < 0.005, i.e. < 0.5%)
  applied to the cell vectors, representing thermal expansion fluctuations.
"""

import numpy as np
import ase.io
from ase import Atoms

# --- Parameters ---
T = 300  # K

# Per-species RMS displacement in Angstrom (sigma = sqrt(U), U in A^2)
# U values are typical crystallographic displacement parameters at 300 K
SIGMA = {
    'O':  np.sqrt(0.015),   # ~0.122 A  (U = 0.015 A^2)
    'Ti': np.sqrt(0.008),   # ~0.089 A  (U = 0.008 A^2)
    'Sr': np.sqrt(0.004),   # ~0.063 A  (U = 0.004 A^2)
}

# Max strain magnitude (dimensionless, i.e. fractional change in lattice vector)
MAX_STRAIN = 0.005  # 0.5%

N_STRUCTURES = 51  # 000 through 050

# --- Read base structure ---
base = ase.io.read('structures/structure_000.cif')
print(f"Base structure: {len(base)} atoms, cell = {base.get_cell()}")

# Verify base is the unperturbed structure (structure_000.cif)
# It should have 5 atoms: 1 Sr, 1 Ti, 3 O
assert len(base) == 5, f"Expected 5 atoms, got {len(base)}"
symbols = base.get_chemical_symbols()
assert symbols.count('Sr') == 1, "Expected 1 Sr"
assert symbols.count('Ti') == 1, "Expected 1 Ti"
assert symbols.count('O') == 3, "Expected 3 O"

# --- Generate perturbed structures ---
np.random.seed(42)  # reproducibility

for n in range(1, N_STRUCTURES):
    atoms = base.copy()
    
    # 1. Apply random lattice strain
    #    Strain tensor: symmetric, small, random
    #    epsilon_ij is the fractional change in the cell
    strain = np.random.uniform(-MAX_STRAIN, MAX_STRAIN, (3, 3))
    # Make symmetric (physical strain is symmetric)
    strain = 0.5 * (strain + strain.T)
    # Apply to cell: new_cell = (I + strain) @ old_cell
    # In ASE, cell vectors are rows, so: new_cell = old_cell @ (I + strain).T
    old_cell = np.array(atoms.get_cell())
    new_cell = old_cell @ (np.eye(3) + strain.T)
    atoms.set_cell(new_cell, scale_atoms=False)  # keep fractional coords fixed
    
    # 2. Apply random thermal displacements
    #    Displacements are in Cartesian coordinates (Angstrom)
    #    Each atom gets a Gaussian displacement with per-species sigma
    positions = np.array(atoms.get_positions())
    for i, sym in enumerate(atoms.get_chemical_symbols()):
        sigma = SIGMA[sym]
        # Displacement in Cartesian A, drawn from N(0, sigma^2) per axis
        disp = np.random.normal(0.0, sigma, 3)
        positions[i] += disp
    atoms.set_positions(positions)
    
    # INVARIANT 1: assert nat and ntyp are consistent with the Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 5, f"Structure {n:03d}: expected nat=5, got {nat}"
    assert ntyp == 3, f"Structure {n:03d}: expected ntyp=3, got {ntyp}"
    
    # Write
    out_path = f'structures/structure_{n:03d}.cif'
    ase.io.write(out_path, atoms)
    
    if n % 10 == 0:
        print(f"  Wrote {out_path} (strain max = {np.max(np.abs(strain)):.5f})")

print(f"\nDone. Wrote structures/structure_001.cif through structure_{N_STRUCTURES-1:03d}.cif")
print(f"Total structures: {N_STRUCTURES} (000 unperturbed + {N_STRUCTURES-1} perturbed)")
