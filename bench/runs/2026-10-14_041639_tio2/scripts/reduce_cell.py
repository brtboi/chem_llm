import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/tio2_base.cif')
print('Original cell: nat =', len(atoms), 'cell =', atoms.cell.array)

kp = standard_kpath(atoms)
print('Space group:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim), 'cell =', prim.cell.array)

# Check space group unchanged
kp2 = standard_kpath(prim)
print('Primitive space group:', kp2['spacegroup'])
assert kp['spacegroup'] == kp2['spacegroup'], 'Space group changed after reduction!'

# Check atom count divides evenly
assert len(atoms) % len(prim) == 0, 'Atom count does not divide evenly!'

ase.io.write('structures/structure_000.cif', prim)
print('Wrote structures/structure_000.cif with', len(prim), 'atoms')
