import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/cu_base.cif')
print('Original cell: nat =', len(atoms), 'symbols:', atoms.get_chemical_symbols())
print('Original cell vectors (A):')
print(atoms.cell)

kp = standard_kpath(atoms)
print('Space group from standard_kpath:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim), 'symbols:', prim.get_chemical_symbols())
print('Primitive cell vectors (A):')
print(prim.cell)

# INVARIANT: reduced atom count divides original evenly
assert len(atoms) % len(prim) == 0, 'Primitive atom count does not divide original'

# Space group number should be unchanged (225)
sg_num = int(kp['spacegroup'].split('#')[-1].strip().rstrip(')'))
assert sg_num == 225, f'Space group changed: {kp["spacegroup"]}'

ase.io.write('structures/structure_000.cif', prim)
print('Wrote structures/structure_000.cif with', len(prim), 'atoms')
