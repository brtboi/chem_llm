import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/base_TiO2.cif')
print('Original cell: nat =', len(atoms))
print('Original cell vectors:')
print(atoms.cell)

kp = standard_kpath(atoms)
print('Space group from standard_kpath:', kp['spacegroup'])

prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim))
print('Primitive cell vectors:')
print(prim.cell)

# Assert space group unchanged (compare the number in the string)
assert '136' in kp['spacegroup'], f"Space group changed: {kp['spacegroup']}"
# Assert reduced atom count divides original evenly
assert len(atoms) % len(prim) == 0, "Reduced atom count does not divide original"

ase.io.write('structures/prim_TiO2.cif', prim)
print('Wrote structures/prim_TiO2.cif')
