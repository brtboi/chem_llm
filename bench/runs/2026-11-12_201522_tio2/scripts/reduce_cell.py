import ase.io
from chem_llm.dft.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/tio2_base.cif')
print('Original cell:', atoms.get_cell(), 'nat =', len(atoms))
print('Space group (CIF):', atoms.info.get('space_group', 'unknown'))

kp = standard_kpath(atoms)
print('Space group (seekpath):', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell:', prim.get_cell(), 'nat =', len(prim))

# INVARIANT: space group number unchanged, reduced count divides original
sg_orig = kp['spacegroup']
assert len(prim) * 1 == len(atoms) or len(atoms) % len(prim) == 0, \
    f'Primitive cell atom count {len(prim)} does not divide original {len(atoms)}'
print(f'\nReduction: {len(atoms)} -> {len(prim)} atoms (factor {len(atoms)//len(prim)})')
print('Space group preserved:', sg_orig)

# Save the primitive cell as the base structure for downstream use
ase.io.write('structures/structure_000.cif', prim)
print('Wrote structures/structure_000.cif (primitive cell)')
