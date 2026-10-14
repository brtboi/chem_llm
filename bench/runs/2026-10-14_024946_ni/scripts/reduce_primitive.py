import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/Ni_base.cif')
print('Original cell:', atoms.get_cell())
print('Original atom count:', len(atoms))
print('Original space group:', standard_kpath(atoms)['spacegroup'])

kp = standard_kpath(atoms)
prim = kp_atoms(kp)
print('Primitive cell:', prim.get_cell())
print('Primitive atom count:', len(prim))
print('Primitive space group:', standard_kpath(prim)['spacegroup'])

# INVARIANT: space group number unchanged, atom count divides evenly
assert '225' in standard_kpath(atoms)['spacegroup'], 'space group changed'
assert len(atoms) % len(prim) == 0, 'atom count does not divide evenly'

ase.io.write('structures/Ni_primitive.cif', prim)
print('Saved primitive cell to structures/Ni_primitive.cif')
