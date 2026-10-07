#!/usr/bin/env python
"""Reduce the base wurtzite ZnO structure to its primitive cell using seekpath.

Per the unit-cell-choice DECISION RULE: prefer seekpath's own reduced cell
(res['primitive_lattice'] / res['primitive_positions'] / res['primitive_types'])
because that is the cell its k-path coordinates refer to.
"""
import numpy as np
import seekpath
from ase.io import read, write
from ase import Atoms
import spglib
import json

# Read the base CIF
atoms = read('structures/zno_base.cif')
print(f"Base structure: {len(atoms)} atoms, cell = {atoms.cell[:].tolist()}")
print(f"Chemical formula: {atoms.get_chemical_formula()}")

# Build the cell tuple for seekpath
cell_tuple = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())

# Get the primitive cell and k-path from seekpath with symprec=0.01
res = seekpath.get_path(cell_tuple, symprec=0.01)

# Verify space group consistency (INVARIANT 3 check)
spglib_spg = spglib.get_symmetry_dataset(cell_tuple, symprec=0.01).number
seekpath_spg = res['spacegroup_number']
print(f"Space group from seekpath: {seekpath_spg}")
print(f"Space group from spglib:   {spglib_spg}")
assert seekpath_spg == spglib_spg, f"Space group mismatch: seekpath={seekpath_spg}, spglib={spglib_spg}"

# Build the primitive cell Atoms object
prim = Atoms(
    cell=res['primitive_lattice'],
    scaled_positions=res['primitive_positions'],
    numbers=res['primitive_types'],
    pbc=True
)
print(f"\nPrimitive cell: {len(prim)} atoms")
print(f"Primitive cell vectors:\n{prim.cell[:].tolist()}")
print(f"Primitive cell positions (Cartesian):\n{prim.get_positions().tolist()}")
print(f"Chemical formula: {prim.get_chemical_formula()}")

# Check that the reduced atom count divides the original evenly
assert len(atoms) % len(prim) == 0, f"Atom count {len(atoms)} not divisible by {len(prim)}"
print(f"\nAtom count: {len(atoms)} -> {len(prim)} (ratio {len(atoms)//len(prim)})")

# Verify space group is unchanged by the reduction
prim_cell_tuple = (prim.cell[:], prim.get_scaled_positions(), prim.get_atomic_numbers())
prim_spg = spglib.get_symmetry_dataset(prim_cell_tuple, symprec=0.01).number
print(f"Primitive cell space group: {prim_spg}")
assert prim_spg == seekpath_spg, f"Space group changed after reduction: {seekpath_spg} -> {prim_spg}"

# Save the primitive cell
write('structures/zno_primitive.cif', prim)
print("\nSaved primitive cell to structures/zno_primitive.cif")

# Save the k-path info for later use.
# seekpath's get_path() returns a dict; the exact key names for labels may vary
# by version, so we handle both 'point_labels' and 'point_labels' alternatives.
# The path is a list of (label_from, label_to) tuples.
path = res['path']
point_coords = res['point_coords']

# Derive labels from the path segments if 'point_labels' is not available
if 'point_labels' in res:
    point_labels = res['point_labels']
else:
    # Build a label list from the path segments
    # Each segment is (label_from, label_to)
    # The full label sequence is: for each segment, label_from, then label_to
    # But consecutive segments share endpoints, so we need to deduplicate
    labels = []
    for i, (l1, l2) in enumerate(path):
        if i == 0:
            labels.append(l1)
        labels.append(l2)
    point_labels = labels

path_info = {
    'path': path,
    'point_coords': point_coords,
    'point_labels': point_labels,
    'spacegroup_number': res['spacegroup_number'],
    'spacegroup_symbol': res.get('spacegroup_symbol', ''),
}
with open('structures/kpath_info.json', 'w') as f:
    json.dump(path_info, f, indent=2, default=str)
print("Saved k-path info to structures/kpath_info.json")
print(f"\nK-path: {path}")
print(f"Point labels: {point_labels}")
print(f"Point coords: {point_coords}")
