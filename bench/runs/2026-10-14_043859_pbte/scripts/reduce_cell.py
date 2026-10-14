import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/base_PbTe.cif')
print('Original cell:', atoms.get_cell())
print('Original atom count:', len(atoms))
print('Original symbols:', atoms.get_chemical_symbols())

kp = standard_kpath(atoms)
print('Space group:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell:', prim.get_cell())
print('Primitive atom count:', len(prim))

# INVARIANT check: space group number unchanged
assert '225' in kp['spacegroup'], f"Space group changed: {kp['spacegroup']}"
# atom count divides evenly
assert len(atoms) % len(prim) == 0, "Primitive atom count does not divide original"

ase.io.write('structures/structure_000.cif', prim)
print('Wrote structures/structure_000.cif with', len(prim), 'atoms')
