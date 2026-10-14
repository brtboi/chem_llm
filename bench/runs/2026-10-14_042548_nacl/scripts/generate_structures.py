#!/usr/bin/env python3
"""
Generate 51 NaCl structures: structure_000.cif (unperturbed base) and
structure_001..050.cif (perturbed copies with random atomic displacements
representative of thermal motion + small random lattice strain).

Temperature choice: 300 K.
  NaCl (rock salt, Fm-3m) is the stable phase of NaCl from 0 K up to its
  melting point (~1074 K). 300 K is well within the stability range and is
  the standard reference temperature for ambient-pressure NaCl phonon data.

Displacement amplitudes (per-axis RMS, in Angstrom):
  The per-axis RMS displacement is sqrt(U) where U is the mean-square
  displacement (crystallographic U in A^2).
  At 300 K, typical Debye-Waller factors for NaCl give:
    Na: U ~ 0.012 A^2  ->  sqrt(U) ~ 0.110 A  (lighter, more mobile)
    Cl: U ~ 0.008 A^2  ->  sqrt(U) ~ 0.089 A  (heavier, less mobile)
  These are representative of thermal motion at 300 K from phonon calculations.
  We use sigma_Na = 0.110 A and sigma_Cl = 0.089 A as the per-axis RMS
  (i.e. the standard deviation of the Gaussian displacement along each axis).

Lattice strain: small random strain tensor with components drawn uniformly
from [-0.01, 0.01] (1% max), representing a small random distortion of the
unit cell.

INVARIANT 1: nat and ntyp are computed from the actual ASE Atoms object,
never hardcoded.
"""

import numpy as np
from ase.io import read, write
from ase import Atoms

# --- Read base structure ---
base = read('structures/NaCl_base.cif')
print(f"Base structure: {len(base)} atoms, {len(set(base.get_chemical_symbols()))} species")
print(f"Cell: {base.cell[:].tolist()}")

# --- Displacement parameters (per-axis RMS in Angstrom) ---
# sigma = sqrt(U) where U is the mean-square displacement in A^2
# Na: U = 0.012 A^2 -> sigma = 0.110 A
# Cl: U = 0.008 A^2 -> sigma = 0.089 A
SIGMA = {'Na': 0.110, 'Cl': 0.089}  # per-axis RMS displacement in Angstrom

# --- Random strain: max 1% per component ---
MAX_STRAIN = 0.01

# --- Seed for reproducibility ---
np.random.seed(42)

# --- INVARIANT 1: compute nat and ntyp from the actual Atoms object ---
nat = len(base)
ntyp = len(set(base.get_chemical_symbols()))
print(f"nat = {nat}, ntyp = {ntyp} (computed from Atoms object)")
assert nat > 0, "Structure has no atoms"
assert ntyp > 0, "Structure has no species"

# --- Generate 51 structures ---
for i in range(51):
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()
        
        # Apply small random lattice strain
        # strain tensor: symmetric 3x3, components in [-MAX_STRAIN, MAX_STRAIN]
        strain = np.random.uniform(-MAX_STRAIN, MAX_STRAIN, (3, 3))
        # Make it symmetric (physical strain tensor is symmetric)
        strain = 0.5 * (strain + strain.T)
        # New cell = (I + strain) @ old_cell
        new_cell = (np.eye(3) + strain) @ base.cell[:]
        atoms.set_cell(new_cell, scale_atoms=False)
        
        # Apply random atomic displacements (thermal motion)
        # Displacements are in Cartesian coordinates (Angstrom)
        # Each atom gets a Gaussian displacement with species-specific sigma
        disp = np.zeros((len(atoms), 3))
        for idx, sym in enumerate(atoms.get_chemical_symbols()):
            sigma = SIGMA[sym]
            disp[idx] = np.random.normal(0.0, sigma, 3)
        atoms.positions += disp
    
    # INVARIANT 1: assert nat/ntyp match the actual structure
    assert len(atoms) == nat, f"Structure {i:03d}: nat mismatch"
    assert len(set(atoms.get_chemical_symbols())) == ntyp, f"Structure {i:03d}: ntyp mismatch"
    
    out_path = f'structures/structure_{i:03d}.cif'
    write(out_path, atoms)
    print(f"Wrote {out_path}")

print("\nDone. Generated 51 structures.")
