import numpy as np
import seekpath
from ase import Atoms
from ase.io import read

# Read the base CIF (conventional cell from Materials Project)
base = read('structures/TiO2_base.cif')
n_base = len(base)
print(f"Base cell: {n_base} atoms, cell = {base.cell[:].tolist()}")

# Build the cell tuple for seekpath (lattice + scaled positions + atomic numbers)
cell_tuple = (base.cell[:], base.get_scaled_positions(), base.get_atomic_numbers())

# seekpath with symprec=0.01 (matches pymatgen default; spglib's 1e-5 default is too tight
# for CIF-round-tripped coordinates and can under-detect symmetry)
res = seekpath.get_path(cell_tuple, symprec=0.01)

# Cross-check the space group number against spglib directly
import spglib
ds = spglib.get_symmetry_dataset(cell_tuple, symprec=0.01)
assert res['spacegroup_number'] == ds.number, (
    f"Space group mismatch: seekpath={res['spacegroup_number']}, spglib={ds.number}")
print(f"Space group: #{res['spacegroup_number']} (seekpath and spglib agree)")

# The primitive cell that seekpath's k-path coordinates refer to
prim = Atoms(
    cell=res['primitive_lattice'],
    scaled_positions=res['primitive_positions'],
    numbers=res['primitive_types'],
    pbc=True,
)
n_prim = len(prim)
print(f"Primitive cell: {n_prim} atoms")

# DECISION RULE check: reduced atom count must divide the original evenly
assert n_base % n_prim == 0, f"{n_prim} does not divide {n_base}"

# Save the primitive cell as the canonical base structure for all downstream steps
from ase.io import write
write('structures/TiO2_primitive.cif', prim)
print(f"Saved primitive cell to structures/TiO2_primitive.cif")
print(f"Atom count: {n_base} (conventional) -> {n_prim} (primitive)")
print(f"Primitive cell vectors:\n{prim.cell[:].tolist()}")
print(f"Atomic numbers: {prim.get_atomic_numbers()}")
print(f"Chemical symbols: {prim.get_chemical_symbols()}")
