import numpy as np
import spglib
from ase.io import read, write

# Read base CIF (generate_cif returned a 12-atom P1 cell, oddly oriented)
atoms = read('structures/base.cif')
n_before = len(atoms)
print('atoms before:', n_before, atoms.get_chemical_formula())

cell_tuple = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
ds_before = spglib.get_symmetry_dataset(cell_tuple, symprec=0.01)
sg_before = ds_before.number if hasattr(ds_before, 'number') else ds_before['number']
print('space group before:', sg_before)
assert sg_before == 136, f'expected 136, got {sg_before}'

# Reduce to primitive cell (rutile primitive = 6 atoms: Ti2 O4)
prim = spglib.standardize_cell(cell_tuple, to_primitive=True, no_idealize=False, symprec=0.01)
plat, ppos, pnum = prim
from ase import Atoms
prim_atoms = Atoms(numbers=pnum, scaled_positions=ppos, cell=plat, pbc=True)
n_after = len(prim_atoms)
print('atoms after:', n_after, prim_atoms.get_chemical_formula())

# Assert space group unchanged and atom count divides evenly
cell_tuple_p = (prim_atoms.cell[:], prim_atoms.get_scaled_positions(), prim_atoms.get_atomic_numbers())
ds_after = spglib.get_symmetry_dataset(cell_tuple_p, symprec=0.01)
sg_after = ds_after.number if hasattr(ds_after, 'number') else ds_after['number']
print('space group after:', sg_after)
assert sg_after == sg_before, 'space group changed by reduction'
assert n_before % n_after == 0, 'reduced count does not divide original'

write('structures/base_primitive.cif', prim_atoms)
print('wrote structures/base_primitive.cif')
