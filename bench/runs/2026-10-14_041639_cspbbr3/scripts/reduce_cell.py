import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/base_CsPbBr3.cif')
n_before = len(atoms)
sg_before = standard_kpath(atoms)['spacegroup']
print('Before reduction:', n_before, 'atoms, space group', sg_before)

kp = standard_kpath(atoms)
prim = kp_atoms(kp)
n_after = len(prim)
sg_after = standard_kpath(prim)['spacegroup']
print('After reduction:', n_after, 'atoms, space group', sg_after)
print('k-path:', kp['path'])

assert n_before % n_after == 0, 'reduced cell must divide original'
assert int(sg_before.split('#')[-1].strip(')')) == int(sg_after.split('#')[-1].strip(')'))

ase.io.write('structures/structure_000.cif', prim)
print('Wrote structures/structure_000.cif (primitive cell)')
