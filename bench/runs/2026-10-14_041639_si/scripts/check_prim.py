from ase.io import read
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = read('structures/base_Si.cif')
print('original nat:', len(atoms))
kp = standard_kpath(atoms)
print('spacegroup:', kp['spacegroup'])
prim = kp_atoms(kp)
print('primitive nat:', len(prim))
print('symbols:', prim.get_chemical_symbols())
print('cell:', prim.cell[:])
assert len(prim) == 2, 'diamond Si primitive cell should have 2 atoms'
assert len(atoms) % len(prim) == 0
print('OK: primitive cell has 2 atoms, original divides evenly')
