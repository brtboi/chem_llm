import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/gan_base.cif')
print('Original cell: nat =', len(atoms), 'symbols:', atoms.get_chemical_symbols())
print('Cell:', atoms.cell.array)

kp = standard_kpath(atoms)
print('Space group from kpath:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim), 'symbols:', prim.get_chemical_symbols())
print('Prim cell:', prim.cell.array)

# INVARIANT: space group number unchanged by reduction
assert '186' in kp['spacegroup'], f"Space group changed: {kp['spacegroup']}"
# Reduced atom count divides original evenly
assert len(atoms) % len(prim) == 0, f"{len(atoms)} not divisible by {len(prim)}"

# Save the primitive cell as the base structure for downstream use
ase.io.write('structures/gan_primitive.cif', prim)
print('Saved structures/gan_primitive.cif')
