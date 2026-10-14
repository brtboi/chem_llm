import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/base.cif')
print('Original cell: nat =', len(atoms), 'symbols:', atoms.get_chemical_symbols())
print('Original space group (from file):', atoms.info.get('space_group', 'unknown'))

kp = standard_kpath(atoms)
print('Space group from standard_kpath:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim), 'symbols:', prim.get_chemical_symbols())

# INVARIANT check: reduced atom count divides original evenly
assert len(atoms) % len(prim) == 0, f"Reduced cell {len(prim)} does not divide original {len(atoms)}"

# Save the primitive cell as the base for downstream use
ase.io.write('structures/base_primitive.cif', prim)
print('Saved primitive cell to structures/base_primitive.cif')
print('Cell vectors (A):')
print(prim.cell.array)
print('Positions (A):')
print(prim.get_positions())
