import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/tio2_base.cif')
print('Original cell: nat =', len(atoms), 'symbols:', atoms.get_chemical_symbols())
print('Original space group (CIF label):', atoms.info.get('space_group', 'n/a'))

kp = standard_kpath(atoms)
print('Space group from standard_kpath:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim), 'symbols:', prim.get_chemical_symbols())

# INVARIANT: reduced atom count divides original evenly
assert len(atoms) % len(prim) == 0, 'Primitive cell atom count does not divide original'
# Space group number must be unchanged (136)
assert '136' in kp['spacegroup'], f'Space group changed after reduction: {kp["spacegroup"]}'

ase.io.write('structures/structure_000.cif', prim)
print('Wrote structures/structure_000.cif with', len(prim), 'atoms')
