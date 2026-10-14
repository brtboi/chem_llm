#!/usr/bin/env python3
"""
Generate 51 perturbed MgO structures (structure_000.cif through structure_050.cif).

Structure 000 is the unperturbed base structure.
Structures 001-050 have:
  - Random atomic displacements (thermal motion)
  - Small random lattice strain

Temperature choice:
  MgO (periclase) is stable at all temperatures up to its melting point (~3100 K).
  We choose T = 300 K (room temperature) as a representative thermal state.
  At 300 K, MgO is well below its Debye temperature (~700 K) and the harmonic
  approximation for thermal displacements is valid.

Displacement amplitudes (RMS, in Angstrom):
  The mean-square displacement U (in A^2) for an atom of mass m at temperature T
  in the harmonic approximation is approximately:
      U = 3*hbar^2 / (m * k_B * T)   (quantum limit, T << Debye T)
  or in the classical limit (T >> Debye T):
      U = 3*k_B*T / (m * omega^2)
  
  For a rough estimate at 300 K:
  - Mg (mass ~24 amu): U_Mg ~ 0.003 A^2  -> RMS = sqrt(U) ~ 0.055 A
  - O  (mass ~16 amu): U_O  ~ 0.0045 A^2 -> RMS = sqrt(U) ~ 0.067 A
  
  These are typical values for ionic crystals at 300 K.
  We use:
      RMS_Mg = 0.055 A  (per axis)
      RMS_O  = 0.067 A  (per axis)
  
  Note: U is the mean-square displacement in A^2; sqrt(U) is the per-axis RMS in A.
  We draw displacements from a Gaussian with sigma = RMS (per axis).

Lattice strain:
  A small random strain tensor (max component ~0.005 = 0.5%) is applied to the
  cell vectors to represent thermal expansion fluctuations.
"""

import numpy as np
import ase.io
from ase import Atoms
from ase.spacegroup import crystal
import os

# Seed for reproducibility
np.random.seed(42)

# Read the base (primitive) MgO structure
base = ase.io.read('structures/prim_MgO.cif')
print(f"Base structure: {len(base)} atoms, symbols: {base.get_chemical_symbols()}")
print(f"Base cell:\n{base.cell}")

# Temperature and displacement parameters (see docstring for justification)
T = 300.0  # K
RMS_Mg = 0.055  # A, per-axis RMS displacement for Mg
RMS_O  = 0.067  # A, per-axis RMS displacement for O

# Strain magnitude (max component of random strain tensor)
STRAIN_MAX = 0.005  # 0.5%

# Number of structures to generate
N_STRUCTURES = 51

# Output directory
os.makedirs('structures', exist_ok=True)

for i in range(N_STRUCTURES):
    atoms = base.copy()
    
    if i == 0:
        # Unperturbed base structure
        pass
    else:
        # Apply random lattice strain
        # Generate a small random symmetric strain tensor
        strain = np.random.uniform(-STRAIN_MAX, STRAIN_MAX, (3, 3))
        # Make it symmetric (strain tensor is symmetric)
        strain = 0.5 * (strain + strain.T)
        # Apply strain to the cell
        new_cell = atoms.cell * (np.eye(3) + strain)
        atoms.set_cell(new_cell, scale_atoms=True)
        
        # Apply random atomic displacements (thermal motion)
        # Displacements are in Cartesian coordinates (Angstrom)
        # Draw from Gaussian with sigma = RMS for each species
        positions = atoms.get_positions().copy()
        symbols = atoms.get_chemical_symbols()
        for j, sym in enumerate(symbols):
            if sym == 'Mg':
                sigma = RMS_Mg
            elif sym == 'O':
                sigma = RMS_O
            else:
                sigma = 0.05  # default
            # Per-axis RMS displacement in A
            disp = np.random.normal(0.0, sigma, 3)
            positions[j] += disp
        atoms.set_positions(positions)
    
    # INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Structure {i:03d}: expected nat=2, got {nat}"
    assert ntyp == 2, f"Structure {i:03d}: expected ntyp=2, got {ntyp}"
    
    # Write the structure
    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)
    
    if i < 3 or i == N_STRUCTURES - 1:
        print(f"Wrote {out_path}: nat={nat}, ntyp={ntyp}")

print(f"\nGenerated {N_STRUCTURES} structures in structures/")
print("Done.")
