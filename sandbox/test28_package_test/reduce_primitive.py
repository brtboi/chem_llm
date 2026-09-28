import numpy as np
import seekpath
import spglib
from ase.io import read, write
from ase import Atoms

# Read the base CIF
atoms = read('structures/base_TiO2.cif')
print(f"Original cell: {atoms.cell}")
print(f"Original atom count: {len(atoms)}")
print(f"Original elements: {atoms.get_chemical_symbols()}")

# Build cell_tuple for seekpath
cell_tuple = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())

# Get path with symprec=0.01 (matching pymatgen's default, not spglib's tight 1e-5)
res = seekpath.get_path(cell_tuple, symprec=0.01)
print(f"Space group number from seekpath: {res['spacegroup_number']}")

# Verify with spglib directly
dataset = spglib.get_symmetry_dataset(cell_tuple, symprec=0.01)
print(f"Space group number from spglib: {dataset.number}")
assert res['spacegroup_number'] == dataset.number, "Space group mismatch between seekpath and spglib!"

# The primitive cell from seekpath
prim = Atoms(
    cell=res['primitive_lattice'],
    scaled_positions=res['primitive_positions'],
    numbers=res['primitive_types'],
    pbc=True
)
print(f"\nPrimitive cell: {prim.cell}")
print(f"Primitive atom count: {len(prim)}")
print(f"Primitive elements: {prim.get_chemical_symbols()}")

# Assert the reduced atom count divides the original evenly
assert len(atoms) % len(prim) == 0, "Reduced atom count does not divide original evenly!"

# Assert space group is unchanged (136 = P4_2/mnm)
assert res['spacegroup_number'] == 136, f"Space group changed after reduction: {res['spacegroup_number']}"

# Save the primitive cell as the base structure for downstream use
write('structures/base_TiO2_primitive.cif', prim)
print(f"\nSaved primitive cell to structures/base_TiO2_primitive.cif")
print(f"\nSummary: {len(atoms)} atoms -> {len(prim)} atoms (primitive cell)")
print(f"k-path: {res['path']}")
print(f"Point coords: {res['point_coords']}")
