import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/NaCl_base.cif')
print('Original cell: nat =', len(atoms), 'spacegroup =', standard_kpath(atoms)['spacegroup'])

kp = standard_kpath(atoms)
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim), 'spacegroup =', kp['spacegroup'])

# INVARIANT: space group number unchanged, atom count divides evenly
assert kp['spacegroup'] == standard_kpath(prim)['spacegroup'], 'space group changed after reduction'
assert len(atoms) % len(prim) == 0, 'primitive cell does not divide original'

ase.io.write('structures/structure_000_base.cif', prim)
print('Saved primitive cell to structures/structure_000_base.cif')
print('nat before:', len(atoms), 'nat after:', len(prim))
