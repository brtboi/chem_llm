import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/SrTiO3_base.cif')
print('Original cell: nat =', len(atoms), 'spacegroup label in CIF:', atoms.info.get('space_group', 'n/a'))

kp = standard_kpath(atoms)
print('Space group from seekpath:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim))

# INVARIANT: space group number unchanged by reduction, and reduced count divides original
assert '221' in kp['spacegroup'], f"space group changed: {kp['spacegroup']}"
assert len(atoms) % len(prim) == 0, "primitive cell does not divide original evenly"

ase.io.write('structures/SrTiO3_primitive.cif', prim)
print('Wrote structures/SrTiO3_primitive.cif with', len(prim), 'atoms')
print('Species:', prim.get_chemical_symbols())
