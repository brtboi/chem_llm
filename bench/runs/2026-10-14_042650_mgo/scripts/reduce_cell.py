import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/MgO_base.cif')
print('Original cell: nat =', len(atoms))
print('Space group from CIF:', atoms.info.get('space_group', 'unknown'))

kp = standard_kpath(atoms)
print('Space group from seekpath:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim))

# INVARIANT: reduced atom count divides original evenly
assert len(atoms) % len(prim) == 0, f"{len(prim)} does not divide {len(atoms)}"

# Space group number should be unchanged
assert '225' in kp['spacegroup'], f"Space group changed: {kp['spacegroup']}"

ase.io.write('structures/MgO_primitive.cif', prim)
print('Wrote structures/MgO_primitive.cif')
print('nat before:', len(atoms), 'nat after:', len(prim))
