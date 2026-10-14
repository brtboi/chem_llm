import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/MoS2_base.cif')
print('Original cell: nat =', len(atoms), 'symbols:', atoms.get_chemical_symbols())
print('Original cell vectors:')
print(atoms.cell)

kp = standard_kpath(atoms)
print('Space group from standard_kpath:', kp['spacegroup'])

prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim), 'symbols:', prim.get_chemical_symbols())
print('Primitive cell vectors:')
print(prim.cell)

# INVARIANT check: reduced atom count divides original evenly
assert len(atoms) % len(prim) == 0, 'Primitive cell atom count does not divide original'

# Space group number check: original was #194 from generate_cif
sg_num = int(kp['spacegroup'].split('#')[-1].strip().rstrip(')'))
assert sg_num == 194, f'Space group changed: {kp["spacegroup"]}'

ase.io.write('structures/structure_000_base.cif', prim)
print('Saved primitive cell to structures/structure_000_base.cif')
print('DONE: original nat =', len(atoms), ', primitive nat =', len(prim))
