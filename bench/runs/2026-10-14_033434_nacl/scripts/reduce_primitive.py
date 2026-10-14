import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/NaCl_base.cif')
print('Original cell: nat =', len(atoms), 'spacegroup =', standard_kpath(atoms)['spacegroup'])

kp = standard_kpath(atoms)
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim), 'spacegroup =', kp['spacegroup'])

# INVARIANT: space group unchanged by reduction, atom count divides evenly
assert kp['spacegroup'] == standard_kpath(prim)['spacegroup'], 'space group changed by reduction'
assert len(atoms) % len(prim) == 0, 'primitive atom count does not divide original'

ase.io.write('structures/NaCl_primitive.cif', prim)
print('Wrote structures/NaCl_primitive.cif')
