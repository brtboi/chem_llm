#!/usr/bin/env python
"""Generate 51 perturbed wurtzite ZnO structures for DFT band-structure calculations.

Structure 000 is the unperturbed primitive cell.
Structures 001-050 have random atomic displacements (thermal motion) and small random lattice strain.

Temperature choice: 300 K (room temperature).
Wurtzite ZnO (alpha-ZnO) is the thermodynamically stable phase at room temperature
and remains stable up to ~1000 C. It is the natural phase found in nature (zincite).

Displacement amplitudes (Debye-Waller U values, mean-square displacements in A^2):
  - Zn: U = 0.005 A^2  -> RMS = sqrt(0.005) = 0.0707 A
  - O:  U = 0.008 A^2  -> RMS = sqrt(0.008) = 0.0894 A
Oxygen is lighter (16 amu vs 65 amu) and more weakly bound in the wurtzite lattice,
so it has a larger mean-square displacement. These U values are representative of
experimental neutron diffraction data for wurtzite ZnO at 300 K.

Lattice strain: small random strain of ~0.5% per axis (Gaussian, sigma=0.005).
This represents the kind of small lattice distortion that occurs in real crystals
due to defects, impurities, and thermal fluctuations.

Units:
  - U values are in A^2 (mean-square displacement)
  - RMS displacement = sqrt(U) in A
  - Strain is dimensionless (fractional change in lattice parameter)
"""
import numpy as np
from ase.io import read, write
from ase import Atoms
import os

# Read the primitive cell
atoms_base = read('structures/zno_primitive.cif')
print(f"Base structure: {len(atoms_base)} atoms")
print(f"Cell:\n{atoms_base.cell[:].tolist()}")
print(f"Chemical formula: {atoms_base.get_chemical_formula()}")

# Debye-Waller U values (mean-square displacements in A^2) at 300 K
# Zn: U = 0.005 A^2, O: U = 0.008 A^2
# RMS displacement = sqrt(U) in A
U_Zn = 0.005  # A^2
U_O = 0.008   # A^2
rms_Zn = np.sqrt(U_Zn)  # A
rms_O = np.sqrt(U_O)    # A
print(f"\nDisplacement amplitudes (RMS = sqrt(U)):")
print(f"  Zn: U = {U_Zn} A^2, RMS = {rms_Zn:.4f} A")
print(f"  O:  U = {U_O} A^2, RMS = {rms_O:.4f} A")

# Random strain: sigma = 0.005 (0.5%)
strain_sigma = 0.005

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory
os.makedirs('structures', exist_ok=True)

# Write structure 000 (unperturbed)
write('structures/structure_000.cif', atoms_base)
print("\nWrote structures/structure_000.cif (unperturbed)")

# Generate perturbed structures 001-050
for i in range(1, 51):
    atoms = atoms_base.copy()
    
    # Apply random lattice strain
    # Strain tensor: small random diagonal strain
    strain = np.random.normal(0, strain_sigma, size=3)
    # Apply strain to cell vectors
    cell = atoms.cell[:].copy()
    for axis in range(3):
        cell[axis] *= (1 + strain[axis])
    atoms.cell = cell
    
    # Apply random atomic displacements
    # Each atom gets a Gaussian displacement with species-specific RMS
    positions = atoms.get_positions().copy()
    for j, atom in enumerate(atoms):
        symbol = atom.symbol
        if symbol == 'Zn':
            rms = rms_Zn
        elif symbol == 'O':
            rms = rms_O
        else:
            rms = 0.05  # default
        # Gaussian displacement in each Cartesian direction
        disp = np.random.normal(0, rms, size=3)
        positions[j] += disp
    atoms.positions = positions
    
    # INVARIANT 1: assert nat/ntyp consistency
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 4, f"Structure {i:03d}: expected 4 atoms, got {nat}"
    assert ntyp == 2, f"Structure {i:03d}: expected 2 species, got {ntyp}"
    
    # Write the structure
    fname = f'structures/structure_{i:03d}.cif'
    write(fname, atoms)
    
    if i <= 3 or i == 50:
        print(f"Wrote {fname}")

print("\nDone. Generated 51 structures (000-050).")
print("Structure 000: unperturbed primitive cell")
print("Structures 001-050: perturbed with thermal displacements + random strain")
