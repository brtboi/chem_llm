import numpy as np
import seekpath
import spglib
from ase.io import read, write
from ase import Atoms

# Read the base CIF (conventional cell from Materials Project)
base = read('structures/TiO2_base.cif')
print(f"Base cell: {len(base)} atoms, cell:\n{base.cell}")

# Build cell_tuple for seekpath
cell_tuple = (base.cell[:], base.get_scaled_positions(), base.get_atomic_numbers())

# Get symmetry dataset with symprec=0.01 (matching pymatgen default, not spglib's tight 1e-5)
dataset = spglib.get_symmetry_dataset(cell_tuple, symprec=0.01)
print(f"spglib space group number: {dataset.number}")
assert dataset.number == 136, f"Expected space group 136, got {dataset.number}"

# Get seekpath result with symprec=0.01
res = seekpath.get_path(cell_tuple, symprec=0.01)
print(f"seekpath space group number: {res['spacegroup_number']}")
assert res['spacegroup_number'] == dataset.number, \
    f"seekpath ({res['spacegroup_number']}) != spglib ({dataset.number})"

# Build the primitive cell from seekpath's own output
prim = Atoms(
    cell=res['primitive_lattice'],
    scaled_positions=res['primitive_positions'],
    numbers=res['primitive_types'],
    pbc=True
)
print(f"\nPrimitive cell: {len(prim)} atoms")
print(f"Primitive cell vectors:\n{prim.cell}")
print(f"Atomic numbers: {prim.get_atomic_numbers()}")
print(f"Chemical symbols: {prim.get_chemical_symbols()}")

# Verify the reduction: primitive atom count should divide the original
assert len(base) % len(prim) == 0, \
    f"Primitive atom count {len(prim)} does not divide original {len(base)}"

# Verify space group is unchanged by reduction
prim_cell_tuple = (prim.cell[:], prim.get_scaled_positions(), prim.get_atomic_numbers())
prim_dataset = spglib.get_symmetry_dataset(prim_cell_tuple, symprec=0.01)
print(f"\nPrimitive cell space group number: {prim_dataset.number}")
assert prim_dataset.number == 136, f"Primitive cell space group changed to {prim_dataset.number}"

# Save the primitive cell as the base structure for downstream use
write('structures/TiO2_primitive.cif', prim)
print("\nSaved primitive cell to structures/TiO2_primitive.cif")

# Print the k-path for reference
print(f"\nK-path: {res['path']}")
print(f"Point coords (fractional in primitive cell):")
# res['path'] is a list of (label_from, label_to) tuples.
# res['point_coords'] is a list of fractional coordinates for each named point.
# The labels in res['path'] correspond to entries in res['point_coords'] by name.
# Build a label->coord dict from the unique labels in the path.
labels_in_path = []
for seg in res['path']:
    for lbl in seg:
        if lbl not in labels_in_path:
            labels_in_path.append(lbl)

# seekpath's res['point_coords'] is indexed by the order of unique labels
# as they appear in the path. We need to map label names to coordinates.
# The standard seekpath API: res['point_coords'] is a list of arrays, one per
# unique high-symmetry point, in the order they appear in res['path']. 
# But the mapping from label name to index is not directly given in older versions.
# Instead, we can use res['path'] which gives (from_label, to_label) pairs, and
# res['point_coords'] which gives coordinates in the same order as the unique
# labels encountered in the path.
# Let's just print the raw data for inspection.
print(f"  Unique labels in path order: {labels_in_path}")
print(f"  point_coords shape: {np.array(res['point_coords']).shape}")
for i, coord in enumerate(res['point_coords']):
    print(f"  point {i}: {coord}")
