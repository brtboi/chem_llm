import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/cdte_base.cif')
print('Original cell: nat =', len(atoms), 'spacegroup =', standard_kpath(atoms)['spacegroup'])

kp = standard_kpath(atoms)
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim), 'spacegroup =', kp['spacegroup'])

# INVARIANT: space group unchanged, atom count divides evenly
assert kp['spacegroup'] == standard_kpath(atoms)['spacegroup'], 'space group changed by reduction'
assert len(atoms) % len(prim) == 0, 'primitive cell does not divide original evenly'

ase.io.write('structures/structure_000.cif', prim)
print('Wrote structures/structure_000.cif with', len(prim), 'atoms')
print('k-path:', kp['path'])
