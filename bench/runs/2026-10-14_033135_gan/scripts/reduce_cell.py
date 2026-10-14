import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/gan_base.cif')
print('Original cell: nat =', len(atoms))
print('Original cell vectors:')
print(atoms.cell)

kp = standard_kpath(atoms)
print('Space group:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim))
print('Primitive cell vectors:')
print(prim.cell)

# Check that the space group is unchanged.
# The spacegroup string has the form 'P6_3mc (#186)' -- extract the number
# between '#' and ')' and compare it to the number generate_cif reported (186).
sg_str = kp['spacegroup']
sg_number = int(sg_str.split('#')[-1].split(')')[0].strip())
assert sg_number == 186, f"Space group changed: {sg_str}"

# Check that the reduced atom count divides the original evenly
assert len(atoms) % len(prim) == 0, "Reduced atom count does not divide original evenly"

print(f"\nAtom count: {len(atoms)} -> {len(prim)}")
print(f"Reduction factor: {len(atoms) // len(prim)}")

# Save the primitive cell for downstream use
ase.io.write('structures/gan_primitive.cif', prim)
print('Saved primitive cell to structures/gan_primitive.cif')
