import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/Al_base.cif')
print('Original cell:', atoms.get_cell())
print('Original nat:', len(atoms))
print('Original spacegroup:', standard_kpath(atoms)['spacegroup'])

kp = standard_kpath(atoms)
prim = kp_atoms(kp)
print('Primitive cell:', prim.get_cell())
print('Primitive nat:', len(prim))
print('Primitive spacegroup:', standard_kpath(prim)['spacegroup'])

# INVARIANT: space group number unchanged, atom count divides evenly
assert standard_kpath(prim)['spacegroup'] == standard_kpath(atoms)['spacegroup'], 'space group changed'
assert len(atoms) % len(prim) == 0, 'atom count does not divide evenly'

ase.io.write('structures/Al_primitive.cif', prim)
print('Saved structures/Al_primitive.cif')
