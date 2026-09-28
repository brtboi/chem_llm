import numpy as np
import spglib
from ase.io import read, write
from ase import Atoms

# Read the base rutile TiO2 CIF from Materials Project (mp-2657, P4_2/mnm #136)
base = read('structures/rutile_TiO2_base.cif')
n_orig = len(base)
print(f"Original cell: {n_orig} atoms")
print(f"Cell:\n{base.cell}")
print(f"Positions (scaled):\n{base.get_scaled_positions()}")

# Build the cell tuple for spglib: (cell, scaled_positions, atomic_numbers)
cell_tuple = (base.cell[:], base.get_scaled_positions(), base.get_atomic_numbers())

# Determine the space group of the original cell (symprec=0.01 matches pymatgen default)
dataset_orig = spglib.get_symmetry_dataset(cell_tuple, symprec=0.01)
sg_orig = dataset_orig.number
print(f"Original space group number: {sg_orig}")
assert sg_orig == 136, f"Expected space group 136, got {sg_orig}"

# Reduce to primitive cell
# spglib.standardize_cell returns a tuple: (cell, scaled_positions, numbers)
prim = spglib.standardize_cell(cell_tuple, to_primitive=True, no_idealize=False, symprec=0.01)
assert prim is not None, "standardize_cell returned None"
prim_cell, prim_pos, prim_types = prim[0], prim[1], prim[2]
n_prim = len(prim_types)
print(f"Primitive cell: {n_prim} atoms")
print(f"Primitive cell vectors:\n{prim_cell}")

# INVARIANT: reduced atom count divides the original evenly
assert n_orig % n_prim == 0, f"{n_prim} does not divide {n_orig}"

# INVARIANT: space group unchanged by the reduction
dataset_prim = spglib.get_symmetry_dataset((prim_cell, prim_pos, prim_types), symprec=0.01)
sg_prim = dataset_prim.number
print(f"Primitive space group number: {sg_prim}")
assert sg_prim == sg_orig, f"Space group changed: {sg_orig} -> {sg_prim}"

# Build the primitive Atoms object
prim_atoms = Atoms(
    numbers=prim_types,
    scaled_positions=prim_pos,
    cell=prim_cell,
    pbc=True,
)
print(f"Primitive atoms: {prim_atoms}")

# Save the primitive base structure for downstream use
write('structures/rutile_TiO2_primitive.cif', prim_atoms)
print("Saved structures/rutile_TiO2_primitive.cif")
print(f"\nSummary: {n_orig} -> {n_prim} atoms, space group {sg_orig} preserved")
