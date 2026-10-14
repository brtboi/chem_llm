import numpy as np
import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

# =============================================================================
# generate_structures.py
# Generates 51 perturbed bcc-Fe structures (structure_000.cif .. structure_050.cif)
# =============================================================================

# --- Temperature choice ---
# bcc-Fe (alpha-Fe) is the stable phase at room temperature and below 912 K.
# We choose T = 300 K, well within the bcc stability range.
# At 300 K, thermal displacements are small but non-zero.

T = 300.0  # Kelvin

# --- Displacement amplitudes ---
# For a single-element system (Fe), we use a typical Debye-Waller factor.
# The mean-square displacement <u^2> for Fe at 300 K is approximately 0.005 A^2.
# This gives an RMS displacement of sqrt(0.005) ~ 0.07 A per axis.
# We use this as the standard deviation for Gaussian displacements.

# U is the mean-square displacement in A^2
# Per-axis RMS displacement = sqrt(U) in A
U_Fe = 0.005  # A^2
rms_Fe = np.sqrt(U_Fe)  # A

# --- Lattice strain ---
# Small random strain to represent thermal expansion fluctuations.
# Typical thermal strain at 300 K is ~1e-4 to 1e-3.
# We use a standard deviation of 1e-3 for the strain tensor components.
strain_std = 1e-3

np.random.seed(42)

# Read the primitive cell
atoms = ase.io.read('structures/Fe_primitive.cif')
print(f"Base structure: {len(atoms)} atoms, cell = {atoms.get_cell()}")

# Verify space group
kp = standard_kpath(atoms)
print(f"Space group: {kp['spacegroup']}")

# Write structure_000 (unperturbed)
ase.io.write('structures/structure_000.cif', atoms)
print("Wrote structure_000.cif (unperturbed)")

# Generate perturbed structures
for i in range(1, 51):
    # Copy the base structure
    pert = atoms.copy()
    
    # Apply random lattice strain
    # Strain tensor: symmetric 3x3 matrix with small random components
    strain = np.random.normal(0, strain_std, (3, 3))
    strain = (strain + strain.T) / 2  # Symmetrize
    
    # Apply strain to the cell
    # new_cell = (I + strain) @ old_cell
    I = np.eye(3)
    new_cell = (I + strain) @ pert.get_cell()
    pert.set_cell(new_cell, scale_atoms=True)
    
    # Apply random atomic displacements
    # Displacements in Cartesian coordinates (Angstrom)
    displacements = np.random.normal(0, rms_Fe, (len(pert), 3))
    pert.positions += displacements
    
    # INVARIANT 1: Assert nat/ntyp consistency
    nat = len(pert)
    ntyp = len(set(pert.get_chemical_symbols()))
    assert nat == 1, f"Expected 1 atom, got {nat}"
    assert ntyp == 1, f"Expected 1 type, got {ntyp}"
    
    # Write the perturbed structure
    filename = f'structures/structure_{i:03d}.cif'
    ase.io.write(filename, pert)
    print(f"Wrote {filename}")

print("\nDone generating 51 structures.")
