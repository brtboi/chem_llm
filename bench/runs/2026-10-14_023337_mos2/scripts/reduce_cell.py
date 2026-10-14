import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/base_MoS2.cif')
print('original cell:', atoms.get_cell())
print('original nat:', len(atoms))
print('original spacegroup label from file:', atoms.info.get('space_group', 'n/a'))

kp = standard_kpath(atoms)
print('spacegroup from seekpath:', kp['spacegroup'])
prim = kp_atoms(kp)
print('primitive nat:', len(prim))
print('primitive cell:', prim.get_cell())

# INVARIANT: space group number unchanged by reduction
assert '194' in kp['spacegroup'], f"space group changed: {kp['spacegroup']}"
# reduced atom count divides original evenly
assert len(atoms) % len(prim) == 0, "reduced cell does not divide original"

ase.io.write('structures/prim_MoS2.cif', prim)
print('wrote structures/prim_MoS2.cif')
