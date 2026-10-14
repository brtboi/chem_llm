import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/MoS2_base.cif')
print('Original cell:', atoms.get_cell())
print('Original atom count:', len(atoms))
print('Original symbols:', atoms.get_chemical_symbols())

kp = standard_kpath(atoms)
print('Space group from kpath:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell:', prim.get_cell())
print('Primitive atom count:', len(prim))
print('Primitive symbols:', prim.get_chemical_symbols())

# Check that the space group number is unchanged
assert '194' in kp['spacegroup'], f"Space group changed: {kp['spacegroup']}"
# Check that the reduced atom count divides the original evenly
assert len(atoms) % len(prim) == 0, f"Atom count {len(atoms)} not divisible by {len(prim)}"

ase.io.write('structures/MoS2_primitive.cif', prim)
print('Saved primitive cell to structures/MoS2_primitive.cif')
