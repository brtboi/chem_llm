#!/usr/bin/env python3
"""
Generate 51 perturbed rutile TiO2 structures (structure_000.cif through structure_050.cif).

Structure 000: unperturbed primitive cell (from structures/primitive_TiO2.cif).
Structures 001-050: random atomic displacements (thermal motion) + small random lattice strain.

Temperature choice: T = 300 K (room temperature).
Justification: Rutile TiO2 is the thermodynamically stable phase at room temperature
and remains stable up to ~1800 K. Above that, it transforms to brookite/anatase.
So 300 K is well within the stability range of rutile.

Thermal displacement amplitudes (Debye-Waller style):
- Crystallographic displacement parameter U is the mean-square displacement in A^2.
- Per-axis RMS displacement = sqrt(U) in A.
- For TiO2 at 300 K, typical B-factor (B = 8*pi^2*U) values from XRD/neutron diffraction:
  * Ti: B ~ 0.05-0.10 A^2  ->  U_Ti = B/(8*pi^2) ~ 0.0006-0.0013 A^2
  * O:  B ~ 0.10-0.20 A^2  ->  U_O  = B/(8*pi^2) ~ 0.0013-0.0025 A^2
  O moves more than Ti because it is lighter (mass 16 vs 48) and more weakly bound.
- I use U_Ti = 0.0010 A^2 (RMS = sqrt(0.0010) = 0.0316 A) and U_O = 0.0020 A^2 (RMS = sqrt(0.0020) = 0.0447 A).
  These are representative of 300 K thermal motion in rutile TiO2.

Lattice strain: small random strain tensor with components ~ N(0, 0.005) (0.5% RMS),
representing thermal expansion fluctuations and small structural distortions.

Units: all displacements in Angstrom, strain dimensionless.
"""

import numpy as np
from ase.io import read, write
from ase import Atoms
import os

# Random seed for reproducibility
np.random.seed(42)

# Read the primitive cell (6 atoms: 2 Ti + 4 O)
base = read('structures/primitive_TiO2.cif')
print(f"Base primitive cell: {len(base)} atoms")
print(f"Elements: {base.get_chemical_symbols()}")
print(f"Cell:\n{base.cell}")

# Verify nat/ntyp invariant for the base structure
nat = len(base)
types = set(base.get_chemical_symbols())
ntyp = len(types)
print(f"nat = {nat}, ntyp = {ntyp} (elements: {sorted(types)})")
assert nat == 6, f"Expected 6 atoms in primitive cell, got {nat}"
assert ntyp == 2, f"Expected 2 element types, got {ntyp}"

# Thermal displacement parameters (mean-square displacement U in A^2)
# Per-axis RMS displacement = sqrt(U) in A
U_Ti = 0.0010  # A^2, RMS = 0.0316 A
U_O = 0.0020   # A^2, RMS = 0.0447 A

# Strain: RMS of each strain component
strain_rms = 0.005  # 0.5% RMS strain

# Number of structures to generate
n_structures = 51

# Output directory
out_dir = 'structures'
os.makedirs(out_dir, exist_ok=True)

for i in range(n_structures):
    if i == 0:
        # Structure 000: unperturbed base
        atoms = base.copy()
    else:
        atoms = base.copy()
        
        # Apply random lattice strain
        # Strain tensor: small random symmetric matrix
        # e_ij = N(0, strain_rms) for i != j, e_ii = N(0, strain_rms)
        strain = np.random.normal(0, strain_rms, (3, 3))
        # Make it symmetric (strain tensor is symmetric)
        strain = 0.5 * (strain + strain.T)
        # New cell = (I + strain) @ old_cell
        new_cell = (np.eye(3) + strain) @ base.cell[:]
        atoms.cell = new_cell
        
        # Apply random atomic displacements (thermal motion)
        # Displacements in Cartesian coordinates (Angstrom)
        # Each atom gets a random displacement drawn from N(0, U_species) per axis
        # U is the mean-square displacement, so std = sqrt(U)
        for idx, atom in enumerate(atoms):
            symbol = atom.symbol
            if symbol == 'Ti':
                U = U_Ti
            elif symbol == 'O':
                U = U_O
            else:
                raise ValueError(f"Unknown element: {symbol}")
            # RMS displacement per axis = sqrt(U)
            rms = np.sqrt(U)
            # Random displacement in Cartesian coordinates (Angstrom)
            disp = np.random.normal(0, rms, 3)
            atoms.positions[idx] += disp
        
        # Ensure positions are within the cell (wrap to [0, 1) in fractional coords)
        # This is important for CIF writing consistency
        atoms.wrap()
    
    # Verify nat/ntyp invariant for each structure
    nat_i = len(atoms)
    ntyp_i = len(set(atoms.get_chemical_symbols()))
    assert nat_i == nat, f"Structure {i:03d}: nat changed from {nat} to {nat_i}"
    assert ntyp_i == ntyp, f"Structure {i:03d}: ntyp changed from {ntyp} to {ntyp_i}"
    
    # Write the structure
    out_path = os.path.join(out_dir, f'structure_{i:03d}.cif')
    write(out_path, atoms)
    
    if i < 3 or i == n_structures - 1:
        print(f"Wrote {out_path}: {nat_i} atoms, ntyp={ntyp_i}")

print(f"\nDone. Generated {n_structures} structures in {out_dir}/")
print(f"Structure 000: unperturbed base")
print(f"Structures 001-050: perturbed (thermal displacements + random strain)")
print(f"T = 300 K, U_Ti = {U_Ti} A^2 (RMS = {np.sqrt(U_Ti):.4f} A), U_O = {U_O} A^2 (RMS = {np.sqrt(U_O):.4f} A)")
print(f"Strain RMS = {strain_rms} ({strain_rms*100:.1f}%)")
