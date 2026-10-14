import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/Al_base.cif')
print('Original cell: nat =', len(atoms), 'spacegroup from CIF metadata may differ')

kp = standard_kpath(atoms)
print('Space group (seekpath):', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim))

# DECISION RULE check: reduced atom count divides original evenly
assert len(atoms) % len(prim) == 0, 'reduced cell does not divide original'
# Space group number unchanged
assert '225' in kp['spacegroup'], f'space group changed: {kp["spacegroup"]}'

ase.io.write('structures/Al_primitive.cif', prim)
print('Wrote structures/Al_primitive.cif with', len(prim), 'atoms')
print('Cell vectors (A):')
print(prim.cell)
