import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/znO_base.cif')
print('Original cell: nat =', len(atoms), 'symbols:', atoms.get_chemical_symbols())
print('Original cell vectors:')
print(atoms.cell)

kp = standard_kpath(atoms)
print('Space group from standard_kpath:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim), 'symbols:', prim.get_chemical_symbols())
print('Primitive cell vectors:')
print(prim.cell)

# INVARIANT: space group unchanged by reduction.
# kp['spacegroup'] is a string like 'P6_3mc (#186)' -- extract the number after '#'.
sg_str = kp['spacegroup']
sg_num = int(sg_str.split('#')[1].split(')')[0].strip())
assert sg_num == 186, f"Space group changed: {sg_str} (number {sg_num})"
# Reduced atom count divides original evenly
assert len(atoms) % len(prim) == 0, f"Atom count {len(atoms)} not divisible by {len(prim)}"

ase.io.write('structures/structure_000.cif', prim)
print('Wrote structures/structure_000.cif with', len(prim), 'atoms')
