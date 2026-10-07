import numpy as np
import seekpath
import spglib
from ase.io import read, write
from ase import Atoms

# Read the base CIF from Materials Project (mp-2815, P6_3/mmc #194)
base = read('structures/MoS2_base.cif')
print(f"Base cell: {len(base)} atoms, cell:\n{base.cell}")
print(f"Elements: {base.get_chemical_symbols()}")

# Build cell_tuple for seekpath (lattice, scaled positions, atomic numbers)
cell_tuple = (base.cell[:], base.get_scaled_positions(), base.get_atomic_numbers())

# Use symprec=0.01 (matching pymatgen's default) to avoid under-detecting symmetry
res = seekpath.get_path(cell_tuple, symprec=0.01)

# Verify space group consistency between seekpath and spglib
spg_dataset = spglib.get_symmetry_dataset(cell_tuple, symprec=0.01)
assert res['spacegroup_number'] == spg_dataset.number, \
    f"Space group mismatch: seekpath={res['spacegroup_number']}, spglib={spg_dataset.number}"
print(f"Space group: #{res['spacegroup_number']} (consistent between seekpath and spglib)")

# Build the primitive cell from seekpath's own reduced cell
prim = Atoms(
    cell=res['primitive_lattice'],
    scaled_positions=res['primitive_positions'],
    numbers=res['primitive_types'],
    pbc=True
)
print(f"Primitive cell: {len(prim)} atoms")
print(f"Primitive cell vectors:\n{prim.cell}")
print(f"Primitive atoms: {prim.get_chemical_symbols()}")

# Assert the reduced atom count divides the original evenly
assert len(base) % len(prim) == 0, \
    f"Reduced atom count {len(prim)} does not divide original {len(base)}"

# Verify space group is unchanged by the reduction
prim_cell_tuple = (prim.cell[:], prim.get_scaled_positions(), prim.get_atomic_numbers())
prim_spg = spglib.get_symmetry_dataset(prim_cell_tuple, symprec=0.01)
assert prim_spg.number == res['spacegroup_number'], \
    f"Space group changed after reduction: {res['spacegroup_number']} -> {prim_spg.number}"

# Save the primitive cell
write('structures/MoS2_primitive.cif', prim)
print(f"\nSaved primitive cell to structures/MoS2_primitive.cif")
print(f"Atom count: {len(base)} (base) -> {len(prim)} (primitive)")

# Print the k-path for reference
print(f"\nK-path: {res['path']}")

# Extract unique labels from the path segments in order of first appearance
# res['path'] is a list of (label_from, label_to) tuples
# res['point_coords'] gives fractional coordinates for each named point in the path
# The labels in point_coords correspond to the unique labels in order of appearance
labels_in_order = []
for seg in res['path']:
    for lbl in seg:
        if lbl not in labels_in_order:
            labels_in_order.append(lbl)

print(f"Point coords (fractional in primitive cell):")
for label, coord in zip(labels_in_order, res['point_coords']):
    print(f"  {label}: {coord}")
