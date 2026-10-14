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
print('Primitive symbols:', prim.get_chemical_symbols())

# INVARIANT 1: nat/ntyp from actual object
nat = len(prim)
ntyp = len(set(prim.get_chemical_symbols()))
print(f'nat={nat}, ntyp={ntyp}')
assert nat == 4, f'Expected 4 atoms in primitive h-BN cell, got {nat}'
assert ntyp == 2, f'Expected 2 species, got {ntyp}'

# Save primitive cell for downstream use
ase.io.write('structures/BN_primitive.cif', prim)
print('Saved structures/BN_primitive.cif')
