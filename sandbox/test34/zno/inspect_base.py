import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/base.cif')
print('Base cell:', atoms.get_cell())
print('Atoms:', len(atoms), 'Species:', sorted(set(atoms.get_chemical_symbols())))

kp = standard_kpath(atoms)
print('Space group:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell:', prim.get_cell())
print('Primitive atoms:', len(prim), 'Species:', sorted(set(prim.get_chemical_symbols())))

# INVARIANT: reduced atom count divides original evenly
assert len(atoms) % len(prim) == 0, 'Primitive atom count must divide original'
print('Reduction factor:', len(atoms) // len(prim))

# Save the primitive cell for downstream use
ase.io.write('structures/prim.cif', prim)
print('Saved structures/prim.cif')
