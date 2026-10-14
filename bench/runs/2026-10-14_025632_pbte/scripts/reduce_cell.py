import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/base_PbTe.cif')
print('original atoms:', len(atoms), 'symbols:', atoms.get_chemical_symbols())
print('original cell:', atoms.cell.array)

kp = standard_kpath(atoms)
print('space group:', kp['spacegroup'])
prim = kp_atoms(kp)
print('primitive atoms:', len(prim), 'symbols:', prim.get_chemical_symbols())
print('primitive cell:', prim.cell.array)

# INVARIANT check: reduced atom count divides original evenly
assert len(atoms) % len(prim) == 0, 'reduced cell does not divide original'
# space group number unchanged
assert '225' in kp['spacegroup'], f'space group changed: {kp["spacegroup"]}'

ase.io.write('structures/prim_PbTe.cif', prim)
print('wrote structures/prim_PbTe.cif')
