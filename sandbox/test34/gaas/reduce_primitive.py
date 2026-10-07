import ase.io
from ase.io import read
from chem_llm.kpath import standard_kpath, kp_atoms

# Read the base CIF (conventional F-43m cell, 8 atoms for zincblende)
base = read('structures/GaAs_base.cif')
print('Base cell: nat =', len(base), 'symbols =', base.get_chemical_symbols())
print('Base cell vectors:')
print(base.cell)

# Get the standard k-path; kp_atoms returns the primitive cell (seekpath's)
kp = standard_kpath(base)
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim), 'symbols =', prim.get_chemical_symbols())
print('Primitive cell vectors:')
print(prim.cell)

# INVARIANT: reduced atom count divides original evenly
assert len(base) % len(prim) == 0, 'Primitive cell atom count does not divide original'

# Save the primitive cell as the base structure for downstream use
ase.io.write('structures/GaAs_primitive.cif', prim)
print('Saved structures/GaAs_primitive.cif')
print('Space group of primitive:', prim.get_space_group() if hasattr(prim, 'get_space_group') else 'N/A')
