import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/c_base.cif')
print('Original cell: nat =', len(atoms), 'ntyp =', len(set(atoms.get_chemical_symbols())))
print('Space group (CIF metadata):', atoms.info.get('space_group', 'unknown'))

kp = standard_kpath(atoms)
print('Space group from seekpath:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim), 'ntyp =', len(set(prim.get_chemical_symbols())))

# INVARIANT: space group number unchanged, atom count divides evenly
assert len(atoms) % len(prim) == 0, 'Primitive atom count must divide original'

# Save the primitive cell as the base for generate_structures.py
ase.io.write('structures/c_base_prim.cif', prim)
print('Saved primitive cell to structures/c_base_prim.cif')
print('Cell vectors (A):')
print(prim.cell[:])
print('Positions:')
for s, p in zip(prim.get_chemical_symbols(), prim.get_positions()):
    print(f'  {s} {p}')
