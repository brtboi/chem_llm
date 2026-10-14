import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/base_MgO.cif')
print('Original cell: nat =', len(atoms), 'symbols:', atoms.get_chemical_symbols())
print('Original space group label in CIF:', atoms.info.get('space_group', 'n/a'))

kp = standard_kpath(atoms)
print('Space group from seekpath:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim), 'symbols:', prim.get_chemical_symbols())

# INVARIANT: space group number unchanged by reduction
assert '225' in kp['spacegroup'], f"Space group changed: {kp['spacegroup']}"
# Reduced atom count divides original evenly
assert len(atoms) % len(prim) == 0, "Primitive cell atom count does not divide original"

ase.io.write('structures/prim_MgO.cif', prim)
print('Wrote structures/prim_MgO.cif')
