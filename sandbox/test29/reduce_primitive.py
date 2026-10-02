import numpy as np
import seekpath
import spglib
from ase.io import read, write
from ase import Atoms

# Read the base CIF (conventional cell from Materials Project)
base = read('structures/TiO2_base.cif')
n_base = len(base)
print(f"Base cell: {n_base} atoms, cell =\n{base.cell.array}")

# Build the cell tuple for seekpath (lattice, scaled positions, atomic numbers)
cell_tuple = (base.cell[:], base.get_scaled_positions(), base.get_atomic_numbers())

# seekpath with symprec=0.01 (matches pymatgen default; spglib's 1e-5 default is too tight
# for CIF-round-tripped coordinates and can under-detect symmetry)
res = seekpath.get_path(cell_tuple, symprec=0.01)

# INVARIANT 3 check: seekpath's space group must agree with spglib's own analysis
sg_spglib = spglib.get_symmetry_dataset(cell_tuple, symprec=0.01).number
assert res['spacegroup_number'] == sg_spglib, (
    f"Space group mismatch: seekpath={res['spacegroup_number']}, spglib={sg_spglib}")
print(f"Space group: #{res['spacegroup_number']} (seekpath and spglib agree)")

# Build the primitive cell from seekpath's own reduced cell -- this is the cell its
# k-path fractional coordinates refer to, so using it makes INVARIANT 3 a no-op.
prim = Atoms(
    cell=res['primitive_lattice'],
    scaled_positions=res['primitive_positions'],
    numbers=res['primitive_types'],
    pbc=True,
)
n_prim = len(prim)
print(f"Primitive cell: {n_prim} atoms, cell =\n{prim.cell.array}")
print(f"Primitive positions (scaled):\n{prim.get_scaled_positions()}")
print(f"Primitive species: {prim.get_chemical_symbols()}")

# DECISION RULE check: reduced atom count must divide the original evenly
assert n_base % n_prim == 0, f"{n_prim} does not divide {n_base}"

# Save the primitive cell as the canonical base structure for all downstream steps
write('structures/TiO2_primitive.cif', prim)
print(f"\nSaved primitive cell to structures/TiO2_primitive.cif")
print(f"Atom count: {n_base} (conventional) -> {n_prim} (primitive)")

# Also print the k-path for reference (used in setup_jobs.py)
print(f"\nk-path: {res['path']}")
print(f"point_coords:\n{res['point_coords']}")
