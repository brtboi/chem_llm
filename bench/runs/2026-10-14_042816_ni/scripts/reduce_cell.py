import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/base_Ni.cif')
print('Original cell: nat =', len(atoms), 'symbols:', atoms.get_chemical_symbols())
print('Original space group (from CIF metadata):', atoms.info.get('space_group', 'unknown'))

kp = standard_kpath(atoms)
print('Space group from seekpath:', kp['spacegroup'])

prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim), 'symbols:', prim.get_chemical_symbols())

# INVARIANT: reduced atom count divides original evenly
assert len(atoms) % len(prim) == 0, f"{len(prim)} does not divide {len(atoms)}"

# Space group number check: original is Fm-3m (#225)
sg_num = int(kp['spacegroup'].split('#')[-1].strip().rstrip(')'))
assert sg_num == 225, f"Space group changed: {kp['spacegroup']}"

print('Reduction factor:', len(atoms) // len(prim))
print('Done. Primitive cell has', len(prim), 'atoms.')
