import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/base_BN.cif')
print('Original cell: nat =', len(atoms), 'symbols:', atoms.get_chemical_symbols())

kp = standard_kpath(atoms)
print('Space group from standard_kpath:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim), 'symbols:', prim.get_chemical_symbols())

# INVARIANT: space group number unchanged by reduction
sg_num = kp['spacegroup'].split('#')[1].strip().strip('()').strip()
assert sg_num == '194', f"Space group changed: {kp['spacegroup']}"
# Reduced atom count divides original evenly
assert len(atoms) % len(prim) == 0, f"{len(prim)} does not divide {len(atoms)}"

ase.io.write('structures/structure_000.cif', prim)
print('Wrote structures/structure_000.cif with', len(prim), 'atoms')
print('K-path:', kp['path'])
