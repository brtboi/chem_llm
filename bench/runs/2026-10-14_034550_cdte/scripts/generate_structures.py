#!/usr/bin/env python
"""
Generate 51 perturbed CdTe structures (structure_000.cif through structure_050.cif).

Structure 000 is the unperturbed primitive cell.
Structures 001-050 have random atomic displacements (thermal motion) and small random lattice strain.

Temperature: 300 K (room temperature)
  - Zincblende CdTe is the stable phase at ambient conditions (300 K, 1 atm).
  - It is a well-known semiconductor used in photovoltaics and IR optics at room temperature.

Displacement amplitudes (crystallographic displacement parameters U):
  - U is the mean-square displacement in A^2.
  - Per-axis RMS displacement = sqrt(U) in A.
  - Cd (Z=48, heavier, more weakly bound): U_Cd = 0.008 A^2 -> RMS = sqrt(0.008) = 0.0894 A
  - Te (Z=52, lighter, more strongly bound): U_Te = 0.005 A^2 -> RMS = sqrt(0.005) = 0.0707 A
  - Justification: Cd is heavier and has a larger ionic radius, so it vibrates with a larger amplitude.
    Te is lighter and more strongly bound in the tetrahedral coordination, so it has a smaller amplitude.
    These values are representative of typical thermal displacements in II-VI semiconductors at 300 K.

Lattice strain:
  - Small random strain tensor (max component ~0.5%) to represent slight lattice distortions.
  - Applied to the cell vectors before displacing atoms.

Units:
  - All displacements are in Angstroms (A).
  - U values are in A^2 (mean-square displacement).
  - RMS = sqrt(U) in A.
"""

import numpy as np
import ase.io
from ase import Atoms
from ase.spacegroup import crystal
import random

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# Read the base primitive cell
base = ase.io.read('structures/structure_000.cif')
print(f"Base structure: {len(base)} atoms, symbols: {base.get_chemical_symbols()}")

# Displacement parameters (U in A^2, RMS = sqrt(U) in A)
U_Cd = 0.008  # A^2
U_Te = 0.005  # A^2
rms_Cd = np.sqrt(U_Cd)  # A
rms_Te = np.sqrt(U_Te)  # A

print(f"Cd: U = {U_Cd} A^2, RMS = {rms_Cd:.4f} A")
print(f"Te: U = {U_Te} A^2, RMS = {rms_Te:.4f} A")

# Maximum strain per component (fractional, ~0.5%)
max_strain = 0.005

for i in range(51):
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()
        
        # Apply small random lattice strain
        # Strain tensor: symmetric, small components
        strain = np.zeros((3, 3))
        for a in range(3):
            for b in range(a, 3):
                strain[a, b] = random.uniform(-max_strain, max_strain)
                strain[b, a] = strain[a, b]
        
        # Apply strain to cell vectors
        # new_cell = (I + strain) @ old_cell
        old_cell = atoms.get_cell()
        new_cell = (np.eye(3) + strain) @ old_cell
        atoms.set_cell(new_cell, scale_atoms=True)
        
        # Apply random atomic displacements (thermal motion)
        # Displacements are in Cartesian coordinates (A)
        # Each atom gets a random displacement from a Gaussian distribution
        # with standard deviation = RMS displacement for that species
        positions = atoms.get_positions().copy()
        symbols = atoms.get_chemical_symbols()
        
        for j, sym in enumerate(symbols):
            if sym == 'Cd':
                rms = rms_Cd
            elif sym == 'Te':
                rms = rms_Te
            else:
                rms = 0.0
            
            # Random displacement in Cartesian coordinates (A)
            disp = np.random.normal(0.0, rms, size=3)
            positions[j] += disp
        
        atoms.set_positions(positions)
    
    # INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Structure {i:03d}: expected 2 atoms, got {nat}"
    assert ntyp == 2, f"Structure {i:03d}: expected 2 species, got {ntyp}"
    
    # Write the structure
    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)
    
    if i == 0 or i % 10 == 0:
        print(f"Wrote {out_path} (nat={nat}, ntyp={ntyp})")

print("\nDone. Generated 51 structures (structure_000.cif through structure_050.cif).")
