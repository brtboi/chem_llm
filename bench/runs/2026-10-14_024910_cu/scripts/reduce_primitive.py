import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/base.cif')
print('Original cell:', atoms.get_cell())
print('Original nat:', len(atoms))

kp = standard_kpath(atoms)
print('Space group:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive nat:', len(prim))
print('Primitive cell:', prim.get_cell())

# INVARIANT: space group number unchanged, atom count divides evenly
orig_sg = 225
prim_sg = int(kp['spacegroup'].split('#')[-1].strip().rstrip(')'))
assert prim_sg == orig_sg, f'Space group changed: {orig_sg} -> {prim_sg}'
assert len(atoms) % len(prim) == 0, 'Primitive atom count does not divide original'

ase.io.write('structures/primitive.cif', prim)
print('Wrote structures/primitive.cif')
