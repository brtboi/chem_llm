import numpy as np
import seekpath
import spglib
from ase.io import read, write
from ase import Atoms

# Read the base CIF from Materials Project (mp-600089, Pm-3m #221)
base = read('structures/CsPbBr3_base.cif')
print(f"Base structure: {len(base)} atoms, cell = {base.cell[:].tolist()}")

# Build cell_tuple for seekpath (lattice + basis aware)
cell_tuple = (base.cell[:], base.get_scaled_positions(), base.get_atomic_numbers())

# Use symprec=0.01 (matching pymatgen's default) to avoid under-detecting symmetry
res = seekpath.get_path(cell_tuple, symprec=0.01)

# Verify space group consistency between seekpath and spglib
spg_dataset = spglib.get_symmetry_dataset(cell_tuple, symprec=0.01)
assert res['spacegroup_number'] == spg_dataset.number, \
    f"Space group mismatch: seekpath={res['spacegroup_number']}, spglib={spg_dataset.number}"
print(f"Space group number: {res['spacegroup_number']} (Pm-3m = 221)")
assert res['spacegroup_number'] == 221, f"Expected Pm-3m (221), got {res['spacegroup_number']}"

# Extract the primitive cell from seekpath's result
prim = Atoms(
    cell=res['primitive_lattice'],
    scaled_positions=res['primitive_positions'],
    numbers=res['primitive_types'],
    pbc=True
)

# INVARIANT: reduced atom count must divide original evenly
n_orig = len(base)
n_prim = len(prim)
assert n_orig % n_prim == 0, f"Primitive atom count {n_prim} does not divide original {n_orig}"
print(f"Atom count: {n_orig} (conventional) -> {n_prim} (primitive)")

# Verify space group is unchanged by the reduction
prim_cell_tuple = (prim.cell[:], prim.get_scaled_positions(), prim.get_atomic_numbers())
prim_spg = spglib.get_symmetry_dataset(prim_cell_tuple, symprec=0.01)
assert prim_spg.number == 221, f"Primitive cell space group changed: {prim_spg.number}"
print(f"Primitive cell space group confirmed: {prim_spg.number} (Pm-3m)")

# Write the primitive cell as the base structure for downstream steps
write('structures/CsPbBr3_primitive.cif', prim)
print(f"Wrote structures/CsPbBr3_primitive.cif with {n_prim} atoms")
print(f"Primitive cell vectors:\n{prim.cell[:].tolist()}")
print(f"Atomic numbers: {prim.get_atomic_numbers()}")
print(f"Chemical symbols: {prim.get_chemical_symbols()}")
