import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/base_Bi2Se3.cif')
print('Original cell:', atoms.get_cell())
print('Original nat:', len(atoms))
print('Original ntyp:', len(set(atoms.get_chemical_symbols())))

kp = standard_kpath(atoms)
print('Space group:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell:', prim.get_cell())
print('Primitive nat:', len(prim))
print('Primitive ntyp:', len(set(prim.get_chemical_symbols())))

# INVARIANT: space group number unchanged, atom count divides evenly
orig_sg = 166
prim_sg_num = int(kp['spacegroup'].split('#')[1].strip().strip(')'))
assert prim_sg_num == orig_sg, f'Space group changed: {orig_sg} -> {prim_sg_num}'
assert len(atoms) % len(prim) == 0, 'Primitive atom count does not divide original'

ase.io.write('structures/primitive_Bi2Se3.cif', prim)
print('Saved primitive cell to structures/primitive_Bi2Se3.cif')
