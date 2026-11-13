import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/tio2_base.cif')
print('Original cell: nat =', len(atoms))
print('Space group from CIF metadata:', atoms.info.get('space_group', 'unknown'))

kp = standard_kpath(atoms)
print('Space group from seekpath:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim))

# INVARIANT: space group number unchanged, atom count divides original
orig_nat = len(atoms)
prim_nat = len(prim)
assert orig_nat % prim_nat == 0, f"{orig_nat} not divisible by {prim_nat}"
print(f"Reduction factor: {orig_nat // prim_nat}")

ase.io.write('structures/structure_000.cif', prim)
print('Wrote structures/structure_000.cif')
print('Primitive cell info:', prim.info)
