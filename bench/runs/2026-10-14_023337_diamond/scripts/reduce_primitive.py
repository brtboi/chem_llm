import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/c_base.cif')
print('conventional cell atoms:', len(atoms))
print('space group (CIF):', atoms.info.get('space_group', 'n/a'))

kp = standard_kpath(atoms)
print('space group (seekpath):', kp['spacegroup'])
prim = kp_atoms(kp)
print('primitive cell atoms:', len(prim))

# DECISION RULE check: space group number unchanged, atom count divides evenly
assert len(atoms) % len(prim) == 0, 'primitive atom count must divide conventional'
assert '227' in kp['spacegroup'], 'space group must remain Fd-3m (#227)'

ase.io.write('structures/structure_000.cif', prim)
print('wrote structures/structure_000.cif (primitive, 2 atoms)')
