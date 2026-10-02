#!/usr/bin/env python3
"""
Generate 51 perturbed rutile TiO2 structures (structure_000.cif through structure_050.cif).

Structure 000 is the unperturbed primitive cell (6 atoms, space group P4_2/mnm #136).
Structures 001-050 have:
  - Random atomic displacements representative of thermal motion at 300 K
  - Small random lattice strain (~1%)

Temperature choice: 300 K (room temperature).
Rutile TiO2 is the thermodynamically stable phase of TiO2 at ambient conditions
(1 atm, 300 K). It remains stable up to very high temperatures (~1800 K) at 1 atm,
so 300 K is well within its stability range. This is the ground-state phase, not
a high-temperature phase like brookite or anatase (which are metastable at RT).

Displacement amplitudes (Debye-Waller U values for rutile TiO2 at 300 K, from
literature neutron/X-ray diffraction studies):
  - Ti: U = 0.002 A^2  -> RMS displacement per axis = sqrt(U) = 0.0447 A
  - O:  U = 0.005 A^2  -> RMS displacement per axis = sqrt(U) = 0.0707 A

Oxygen moves more than titanium because it is lighter (16 vs 48 amu) and the
Ti-O bond has a softer transverse mode for O. The U values are mean-square
displacements in A^2; the per-axis RMS displacement is sqrt(U) in A.

Lattice strain: a small random strain tensor with components drawn uniformly
from [-0.01, 0.01] (i.e. up to 1% strain in each direction). This represents
small deviations from the ideal lattice due to thermal expansion fluctuations
and local strain fields.

Units:
  - U (Debye-Waller parameter): A^2 (mean-square displacement)
  - RMS displacement: A (root-mean-square, per axis)
  - Strain: dimensionless (fractional change in lattice parameter)
  - Cell vectors: A (Angstroms)
  - Scaled positions: dimensionless (fractional coordinates)
"""

import numpy as np
from ase.io import read, write
from ase import Atoms
import os

# Random seed for reproducibility
np.random.seed(42)

# Read the primitive cell (6 atoms, P4_2/mnm #136)
base = read('structures/TiO2_primitive.cif')
print(f"Base primitive cell: {len(base)} atoms")
print(f"Cell:\n{base.cell}")
print(f"Chemical symbols: {base.get_chemical_symbols()}")

# Verify nat/ntyp INVARIANT for the base structure
nat_base = len(base)
types_base = set(base.get_chemical_symbols())
ntyp_base = len(types_base)
print(f"nat = {nat_base}, ntyp = {ntyp_base} (types: {sorted(types_base)})")
assert nat_base == 6, f"Expected 6 atoms in primitive cell, got {nat_base}"
assert ntyp_base == 2, f"Expected 2 element types, got {ntyp_base}"

# Debye-Waller U values (A^2) for rutile TiO2 at 300 K
# These are mean-square displacements; RMS per axis = sqrt(U) in A
U_Ti = 0.002  # A^2
U_O = 0.005   # A^2

# RMS displacement per axis (A)
rms_Ti = np.sqrt(U_Ti)  # ~0.0447 A
rms_O = np.sqrt(U_O)    # ~0.0707 A

print(f"\nDisplacement parameters (300 K):")
print(f"  Ti: U = {U_Ti} A^2, RMS = {rms_Ti:.4f} A")
print(f"  O:  U = {U_O} A^2, RMS = {rms_O:.4f} A")

# Strain magnitude: up to 1% in each direction
strain_max = 0.01

# Output directory
out_dir = 'structures'
os.makedirs(out_dir, exist_ok=True)

# Structure 000: unperturbed base
write(os.path.join(out_dir, 'structure_000.cif'), base)
print(f"\nWrote structure_000.cif (unperturbed)")

# Structures 001-050: perturbed
for n in range(1, 51):
    atoms = base.copy()
    
    # --- Apply random lattice strain ---
    # Generate a small random strain tensor (symmetric, trace-free for volume preservation
    # is not required here; we allow small volume changes too)
    strain = np.random.uniform(-strain_max, strain_max, size=(3, 3))
    # Make it symmetric (physical strain tensor is symmetric)
    strain = 0.5 * (strain + strain.T)
    
    # Apply strain to the cell: new_cell = (I + strain) @ old_cell
    # In ASE, cell vectors are rows, so: new_cell = (I + strain.T) @ old_cell
    # Actually, if we think of the cell as a matrix C where rows are cell vectors,
    # and strain as a tensor that acts on the right (deforming the lattice),
    # then new_cell = C @ (I + strain).T
    # Let's use the standard: new_cell = (I + strain) @ C if C is column vectors.
    # ASE uses row vectors, so: new_cell = C @ (I + strain).T
    I = np.eye(3)
    new_cell = atoms.cell @ (I + strain).T
    atoms.cell = new_cell
    
    # --- Apply random atomic displacements ---
    # Displacements are in Cartesian coordinates (A), drawn from a Gaussian
    # with standard deviation = RMS displacement for that species.
    # This is equivalent to a Debye-Waller factor exp(-2*pi^2 * U * |G|^2)
    # in the structure factor, where U is the mean-square displacement.
    
    symbols = atoms.get_chemical_symbols()
    numbers = atoms.get_atomic_numbers()
    
    # Get current Cartesian positions
    pos = atoms.get_positions().copy()
    
    for i, (sym, num) in enumerate(zip(symbols, numbers)):
        if sym == 'Ti':
            rms = rms_Ti
        elif sym == 'O':
            rms = rms_O
        else:
            raise ValueError(f"Unknown element: {sym}")
        
        # Draw Gaussian displacement in each Cartesian direction
        # Standard deviation = RMS displacement (A)
        disp = np.random.normal(0.0, rms, size=3)
        pos[i] += disp
    
    # Set new positions (Cartesian, A)
    atoms.positions = pos
    
    # Verify nat/ntyp INVARIANT for this perturbed structure
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 6, f"Structure {n:03d}: expected 6 atoms, got {nat}"
    assert ntyp == 2, f"Structure {n:03d}: expected 2 element types, got {ntyp}"
    
    # Verify positions are within reasonable bounds (not displaced too far)
    # Max displacement should be ~3*sigma = 3*0.07 = 0.21 A for O
    # Check that no atom is displaced more than 0.5 A from its original position
    orig_pos = base.get_positions()
    # Note: the cell has changed, so we need to compare in the same frame.
    # For a sanity check, just verify positions are finite and reasonable.
    assert np.all(np.isfinite(atoms.get_positions())), \
        f"Structure {n:03d}: non-finite positions"
    
    # Write the structure
    fname = os.path.join(out_dir, f'structure_{n:03d}.cif')
    write(fname, atoms)
    
    if n % 10 == 0:
        print(f"Wrote structure_{n:03d}.cif (strain max = {np.max(np.abs(strain)):.4f})")

print(f"\nDone. Wrote 51 structures to {out_dir}/")
print("Structure 000: unperturbed primitive cell")
print("Structures 001-050: perturbed with thermal displacements + random strain")
