import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/base_C.cif')
print('original cell:', atoms.get_cell())
print('original nat:', len(atoms))

kp = standard_kpath(atoms)
print('spacegroup:', kp['spacegroup'])
prim = kp_atoms(kp)
print('primitive nat:', len(prim))

# INVARIANT: space group unchanged by reduction, atom count divides evenly
# kp['spacegroup'] is a string like 'P6_3/mmc (#194)' -- compare the number, not the symbol
sg_num = kp['spacegroup'].split('#')[1].strip().rstrip(')')
assert sg_num == '194', kp['spacegroup']
assert len(atoms) % len(prim) == 0

ase.io.write('structures/primitive_C.cif', prim)
print('wrote structures/primitive_C.cif')
