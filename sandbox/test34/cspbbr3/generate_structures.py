#!/usr/bin/env python3
"""
Generate 51 perturbed CsPbBr3 structures (structure_000.cif through structure_050.cif).

Structure 000: unperturbed primitive cell (from structures/CsPbBr3_primitive.cif).
Structures 001-050: random atomic displacements (thermal motion) + small random lattice strain.

Temperature choice: T = 300 K
  Cubic CsPbBr3 (Pm-3m) is the high-temperature phase, stable from ~290 K to ~400 K.
  Below ~290 K it transforms to orthorhombic (Pmna). 300 K is safely within the cubic
  stability range and corresponds to room-temperature conditions.

Displacement amplitudes (Debye-Waller style):
  The crystallographic displacement parameter U is the mean-square displacement in A^2.
  The per-axis RMS displacement is sqrt(U) in A.
  We draw each atom's displacement from a Gaussian with sigma = sqrt(U) per axis.

  Species-specific U values at 300 K (typical for cubic CsPbBr3):
    Cs: U = 0.05 A^2  -> RMS = sqrt(0.05) ~ 0.224 A
         Cs is the lightest atom (Z=55) and sits on the A-site of the perovskite,
         which is the most weakly bound and most mobile site. It moves the most.
    Pb: U = 0.01 A^2  -> RMS = sqrt(0.01) ~ 0.100 A
         Pb is the heaviest atom (Z=82) and sits at the B-site center, strongly
         bonded to the Br octahedron. It moves the least.
    Br: U = 0.02 A^2  -> RMS = sqrt(0.02) ~ 0.141 A
         Br (Z=35) is intermediate in mass and sits at the corner of the octahedron.

Lattice strain: small random strain of up to ~1% per axis (diagonal strain tensor),
  representative of thermal expansion/contraction fluctuations.
"""

import numpy as np
from ase.io import read, write
from ase import Atoms
import os

# --- Parameters ---
T = 300.0  # K, within cubic CsPbBr3 stability range (~290-400 K)

# Per-species mean-square displacement U in A^2 (crystallographic displacement parameter)
# RMS per-axis displacement = sqrt(U) in A
U_Cs = 0.05   # A^2 -> RMS ~ 0.224 A (lightest, A-site, most mobile)
U_Pb = 0.01   # A^2 -> RMS ~ 0.100 A (heaviest, B-site, least mobile)
U_Br = 0.02   # A^2 -> RMS ~ 0.141 A (intermediate mass, octahedron corner)

# Map element symbol -> U value (A^2)
U_map = {'Cs': U_Cs, 'Pb': U_Pb, 'Br': U_Br}

# Maximum random strain per axis (fractional, ~1%)
max_strain = 0.01

# Number of structures to generate (000 through 050)
n_structures = 51

# Seed for reproducibility
np.random.seed(42)

# --- Read base primitive cell ---
base = read('structures/CsPbBr3_primitive.cif')
print(f"Base primitive cell: {len(base)} atoms, {len(set(base.get_chemical_symbols()))} species")
print(f"Cell:\n{base.cell}")
print(f"Elements: {sorted(set(base.get_chemical_symbols()))}")

# Verify all elements have a U value
for sym in set(base.get_chemical_symbols()):
    assert sym in U_map, f"No U value defined for element {sym}"

# --- Generate structures ---
for i in range(n_structures):
    if i == 0:
        # Structure 000: unperturbed base
        atoms = base.copy()
    else:
        atoms = base.copy()
        
        # Apply small random lattice strain (diagonal strain tensor, up to ~1% per axis)
        # strain_ij = (a_i' - a_i) / a_i, so new cell = old cell * (1 + strain)
        strain = np.random.uniform(-max_strain, max_strain, size=3)
        new_cell = base.cell.copy()
        for axis in range(3):
            new_cell[axis] = base.cell[axis] * (1.0 + strain[axis])
        atoms.cell = new_cell
        
        # Apply random atomic displacements (thermal motion)
        # Each atom gets a Gaussian displacement with sigma = sqrt(U) per axis
        # Displacements are in Cartesian coordinates (A)
        positions = atoms.get_positions().copy()
        for j, sym in enumerate(atoms.get_chemical_symbols()):
            U = U_map[sym]  # A^2
            sigma = np.sqrt(U)  # A, per-axis RMS displacement
            disp = np.random.normal(0.0, sigma, size=3)  # A
            positions[j] += disp
        atoms.positions = positions
    
    # INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 5, f"Structure {i:03d}: expected nat=5, got {nat}"
    assert ntyp == 3, f"Structure {i:03d}: expected ntyp=3, got {ntyp}"
    
    # Write CIF
    out_path = f'structures/structure_{i:03d}.cif'
    write(out_path, atoms)
    
    if i < 3 or i == n_structures - 1:
        print(f"Wrote {out_path}: nat={nat}, ntyp={ntyp}")

print(f"\nDone. Generated {n_structures} structures in structures/")
print(f"Temperature: {T} K (cubic CsPbBr3 stable ~290-400 K)")
print(f"Displacement amplitudes (U in A^2, RMS=sqrt(U) in A):")
print(f"  Cs: U={U_Cs} A^2, RMS={np.sqrt(U_Cs):.4f} A")
print(f"  Pb: U={U_Pb} A^2, RMS={np.sqrt(U_Pb):.4f} A")
print(f"  Br: U={U_Br} A^2, RMS={np.sqrt(U_Br):.4f} A")
print(f"Max strain per axis: {max_strain*100:.1f}%")
