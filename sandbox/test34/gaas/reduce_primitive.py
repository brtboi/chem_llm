import numpy as np
import seekpath
import spglib
from ase.io import read, write
from ase import Atoms

base = read('structures/GaAs_base.cif')
print('Base cell: nat =', len(base), 'cell =\n', base.cell[:])

cell_tuple = (base.cell[:], base.get_scaled_positions(), base.get_atomic_numbers())
res = seekpath.get_path(cell_tuple, symprec=0.01)
# seekpath's result dict keys vary by version; only spacegroup_number is guaranteed.
print('seekpath space group number:', res['spacegroup_number'])

# INVARIANT 3 check: seekpath space group must match spglib's own analysis at the same symprec
spg = spglib.get_symmetry_dataset(cell_tuple, symprec=0.01)
assert res['spacegroup_number'] == spg.number, (res['spacegroup_number'], spg.number)
# The base structure is unperturbed and generate_cif reported F-43m (#216) as authoritative
assert res['spacegroup_number'] == 216, res['spacegroup_number']

prim = Atoms(cell=res['primitive_lattice'], scaled_positions=res['primitive_positions'],
             numbers=res['primitive_types'], pbc=True)
print('Primitive cell: nat =', len(prim), 'cell =\n', prim.cell[:])
print('Primitive symbols:', prim.get_chemical_symbols())

# DECISION RULE check: reduction must preserve space group and atom count must divide evenly
prim_tuple = (prim.cell[:], prim.get_scaled_positions(), prim.get_atomic_numbers())
spg_prim = spglib.get_symmetry_dataset(prim_tuple, symprec=0.01)
assert spg_prim.number == 216, spg_prim.number
assert len(base) % len(prim) == 0, (len(base), len(prim))

write('structures/GaAs_primitive.cif', prim)
print('Wrote structures/GaAs_primitive.cif')
print('Path:', res['path'])
print('Point coords:', res['point_coords'])
