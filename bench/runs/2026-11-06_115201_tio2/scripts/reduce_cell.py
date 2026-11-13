import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/structure_000_base.cif')
n_before = len(atoms)
print('atoms before reduction:', n_before)

kp = standard_kpath(atoms)
print('space group:', kp['spacegroup'])
prim = kp_atoms(kp)
n_after = len(prim)
print('atoms after reduction:', n_after)

# DECISION RULE check: space group unchanged, reduced count divides original
g_before = int(atoms.get_space_group().number()) if hasattr(atoms, 'get_space_group') else None
assert '136' in kp['spacegroup'], kp['spacegroup']
assert n_before % n_after == 0, (n_before, n_after)

ase.io.write('structures/structure_000.cif', prim)
print('wrote structures/structure_000.cif with', n_after, 'atoms')
print(prim.get_chemical_symbols())
