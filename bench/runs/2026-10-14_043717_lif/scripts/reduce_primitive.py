import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/LiF_base.cif')
print('Original cell: nat =', len(atoms), 'symbols:', atoms.get_chemical_symbols())
print('Original space group (from standard_kpath):', standard_kpath(atoms)['spacegroup'])

kp = standard_kpath(atoms)
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim), 'symbols:', prim.get_chemical_symbols())
print('Primitive space group:', standard_kpath(prim)['spacegroup'])

# INVARIANT check: space group number unchanged, atom count divides evenly
assert len(atoms) % len(prim) == 0, 'primitive atom count must divide original'
assert standard_kpath(atoms)['spacegroup'] == standard_kpath(prim)['spacegroup'], 'space group must be unchanged'

ase.io.write('structures/LiF_primitive.cif', prim)
print('Wrote structures/LiF_primitive.cif')
print('k-path:', kp['path'])
