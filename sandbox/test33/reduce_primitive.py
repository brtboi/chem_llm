import numpy as np
import seekpath
import spglib
from ase.io import read
from ase import Atoms

# Read the base CIF from Materials Project (mp-2657, P4_2/mnm #136)
base = read('structures/base_TiO2.cif')
print(f"Base structure: {len(base)} atoms, cell:\n{base.cell}")
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

# Extract the primitive cell from seekpath's result
prim = Atoms(
    cell=res['primitive_lattice'],
    scaled_positions=res['primitive_positions'],
    numbers=res['primitive_types'],
    pbc=True
)
print(f"\nPrimitive cell: {len(prim)} atoms")
print(f"Primitive cell vectors:\n{prim.cell}")
print(f"Primitive positions (scaled):\n{prim.get_scaled_positions()}")
print(f"Primitive elements: {prim.get_chemical_symbols()}")

# Verify the reduction: primitive atom count should divide original evenly
assert len(base) % len(prim) == 0, \
    f"Primitive atom count {len(prim)} does not divide original {len(base)}"
print(f"\nReduction factor: {len(base)}/{len(prim)} = {len(base)//len(prim)}")

# Verify space group is unchanged by the reduction
prim_cell_tuple = (prim.cell[:], prim.get_scaled_positions(), prim.get_atomic_numbers())
prim_spg = spglib.get_symmetry_dataset(prim_cell_tuple, symprec=0.01)
assert prim_spg.number == res['spacegroup_number'], \
    f"Space group changed after reduction: {res['spacegroup_number']} -> {prim_spg.number}"
print(f"Space group unchanged after reduction: #{prim_spg.number}")

# Save the primitive cell for downstream use
prim.write('structures/primitive_TiO2.cif')
print("\nSaved primitive cell to structures/primitive_TiO2.cif")

# Record the k-path for reference
print(f"\nK-path: {res['path']}")
print(f"Point coords (fractional in primitive cell):")
# seekpath's get_path returns 'point_coords' as a DICT keyed by label string
# (e.g. {'GAMMA': [0,0,0], 'X': [0.5,0,0], ...}), not a list.
# Access each label directly from the dict.
for seg in res['path']:
    for lbl in seg:
        coord = res['point_coords'][lbl]
        print(f"  {lbl}: {coord}")
