import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/cdte_base.cif')
print('Original cell: nat =', len(atoms), 'ntyp =', len(set(atoms.get_chemical_symbols())))
print('Space group from standard_kpath:', standard_kpath(atoms)['spacegroup'])

kp = standard_kpath(atoms)
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim), 'ntyp =', len(set(prim.get_chemical_symbols())))
print('Space group of primitive cell:', standard_kpath(prim)['spacegroup'])

# Check that the space group number is unchanged
assert '216' in standard_kpath(atoms)['spacegroup'], 'Space group changed!'
assert '216' in standard_kpath(prim)['spacegroup'], 'Space group changed!'

# Check that reduced atom count divides original evenly
assert len(atoms) % len(prim) == 0, 'Atom count does not divide evenly!'

ase.io.write('structures/structure_000.cif', prim)
print('Wrote structures/structure_000.cif with', len(prim), 'atoms')
print('K-path:', kp['path'])
