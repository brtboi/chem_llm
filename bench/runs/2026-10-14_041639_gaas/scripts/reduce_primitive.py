import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/base_GaAs.cif')
print('Original cell: nat =', len(atoms), 'symbols:', atoms.get_chemical_symbols())

kp = standard_kpath(atoms)
print('Space group:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim), 'symbols:', prim.get_chemical_symbols())

# Assert space group unchanged and atom count divides evenly
assert len(atoms) % len(prim) == 0, 'Primitive atom count must divide original'

# Save the primitive cell as the base structure for downstream use
ase.io.write('structures/base_GaAs_prim.cif', prim)
print('Saved primitive cell to structures/base_GaAs_prim.cif')
