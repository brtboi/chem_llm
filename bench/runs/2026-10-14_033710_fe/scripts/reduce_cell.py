import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/Fe_base.cif')
print('Original cell:', atoms.get_cell())
print('Original nat:', len(atoms))
print('Space group from file:', atoms.info.get('space_group', 'unknown'))

kp = standard_kpath(atoms)
print('Space group from seekpath:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell:', prim.get_cell())
print('Primitive nat:', len(prim))

# Assert space group number unchanged
sg_orig = kp['spacegroup']
print('Space group:', sg_orig)

# Save primitive cell as structure_000 base
ase.io.write('structures/Fe_primitive.cif', prim)
print('Saved primitive cell to structures/Fe_primitive.cif')
