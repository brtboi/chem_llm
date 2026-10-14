import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/cu_base.cif')
print('Original cell: nat =', len(atoms), 'ntyp =', len(set(atoms.get_chemical_symbols())))
print('Cell:', atoms.cell.array)

kp = standard_kpath(atoms)
print('Space group:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim), 'ntyp =', len(set(prim.get_chemical_symbols())))
print('Prim cell:', prim.cell.array)

# INVARIANT: space group number unchanged by reduction
assert '225' in kp['spacegroup'], f"Space group changed: {kp['spacegroup']}"
# Reduced atom count divides original evenly
assert len(atoms) % len(prim) == 0, "Reduced atom count does not divide original"

ase.io.write('structures/cu_primitive.cif', prim)
print('Saved primitive cell to structures/cu_primitive.cif')
