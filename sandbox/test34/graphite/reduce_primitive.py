import numpy as np
import seekpath
import spglib
from ase.io import read
from ase import Atoms

# Read the base CIF
atoms = read('structures/base_C.cif')
print(f"Original cell: {atoms.cell}")
print(f"Original atom count: {len(atoms)}")
print(f"Original species: {atoms.get_chemical_symbols()}")

# Build cell_tuple for seekpath
cell_tuple = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())

# Get path with symprec=0.01 (matching pymatgen default, avoiding spglib's tight 1e-5)
res = seekpath.get_path(cell_tuple, symprec=0.01)
print(f"Space group number from seekpath: {res['spacegroup_number']}")

# Verify with spglib directly
spg_dataset = spglib.get_symmetry_dataset(cell_tuple, symprec=0.01)
print(f"Space group number from spglib: {spg_dataset.number}")
assert res['spacegroup_number'] == spg_dataset.number, "Space group mismatch between seekpath and spglib"

# The primitive cell from seekpath
prim_cell = res['primitive_lattice']
prim_pos = res['primitive_positions']
prim_types = res['primitive_types']
print(f"\nPrimitive cell: {prim_cell}")
print(f"Primitive atom count: {len(prim_types)}")
print(f"Primitive species: {prim_types}")

# Assert the reduced atom count divides the original evenly
assert len(atoms) % len(prim_types) == 0, "Reduced atom count does not divide original evenly"

# Build the primitive Atoms object
prim = Atoms(cell=prim_cell, scaled_positions=prim_pos, numbers=prim_types, pbc=True)
print(f"\nPrimitive Atoms object:")
print(f"  nat = {len(prim)}")
print(f"  ntyp = {len(set(prim.get_chemical_symbols()))}")
print(f"  cell = {prim.cell}")
print(f"  positions = {prim.get_positions()}")

# Save the primitive cell for downstream use
prim.write('structures/primitive_C.cif')
print("\nSaved primitive cell to structures/primitive_C.cif")

# Print the k-path for reference
print(f"\nK-path: {res['path']}")
print(f"Point coords: {res['point_coords']}")
