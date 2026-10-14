import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/cdte_base.cif')
print('Original cell:', atoms.get_cell())
print('Original atom count:', len(atoms))
print('Original symbols:', atoms.get_chemical_symbols())

kp = standard_kpath(atoms)
print('Space group from kpath:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell:', prim.get_cell())
print('Primitive atom count:', len(prim))
print('Primitive symbols:', prim.get_chemical_symbols())

# INVARIANT: space group number unchanged, reduced count divides original
assert '216' in kp['spacegroup'], f"Space group changed: {kp['spacegroup']}"
assert len(atoms) % len(prim) == 0, "Reduced atom count does not divide original"

ase.io.write('structures/structure_000.cif', prim)
print('Wrote structures/structure_000.cif')
