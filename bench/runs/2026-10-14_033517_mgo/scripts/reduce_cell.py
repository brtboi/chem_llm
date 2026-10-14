import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/base_MgO.cif')
print('original cell:', atoms.get_cell())
print('original nat:', len(atoms))
print('original symbols:', atoms.get_chemical_symbols())

kp = standard_kpath(atoms)
print('space group:', kp['spacegroup'])
prim = kp_atoms(kp)
print('primitive nat:', len(prim))
print('primitive cell:', prim.get_cell())

# INVARIANT: space group number unchanged by reduction
assert '225' in kp['spacegroup'], f'space group changed: {kp["spacegroup"]}'
# reduced atom count divides original evenly
assert len(atoms) % len(prim) == 0, 'primitive count does not divide original'

ase.io.write('structures/prim_MgO.cif', prim)
print('wrote structures/prim_MgO.cif')
