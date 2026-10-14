import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/Ni_base.cif')
print('Original cell:', atoms.get_cell())
print('Original nat:', len(atoms))
print('Space group from file:', atoms.info.get('space_group', 'unknown'))

kp = standard_kpath(atoms)
print('Space group from seekpath:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell:', prim.get_cell())
print('Primitive nat:', len(prim))

# INVARIANT: space group number unchanged by reduction
orig_sg = 225  # from generate_cif
assert '225' in kp['spacegroup'], f'Space group changed: {kp["spacegroup"]}'
assert len(atoms) % len(prim) == 0, 'Reduced atom count does not divide original'

prim.write('structures/structure_000.cif')
print('Wrote structures/structure_000.cif')
