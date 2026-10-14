import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/BN_base.cif')
print('Original cell:', atoms.get_cell())
print('Original nat:', len(atoms))
print('Original symbols:', atoms.get_chemical_symbols())

kp = standard_kpath(atoms)
print('Space group:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive nat:', len(prim))
print('Primitive cell:', prim.get_cell())

# INVARIANT: space group number unchanged by reduction
assert '194' in kp['spacegroup'], f"Space group changed: {kp['spacegroup']}"
# Reduced atom count must divide original evenly
assert len(atoms) % len(prim) == 0, "Primitive cell does not divide original"

ase.io.write('structures/BN_primitive.cif', prim)
print('Saved primitive cell to structures/BN_primitive.cif')
