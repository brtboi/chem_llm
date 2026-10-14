import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/base_PbTe.cif')
print('Original cell: nat =', len(atoms), 'symbols:', atoms.get_chemical_symbols())
print('Original space group (from standard_kpath):', standard_kpath(atoms)['spacegroup'])

kp = standard_kpath(atoms)
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim), 'symbols:', prim.get_chemical_symbols())
print('Primitive space group:', standard_kpath(prim)['spacegroup'])

# Check space group number unchanged
sg_orig = standard_kpath(atoms)['spacegroup']
sg_prim = standard_kpath(prim)['spacegroup']
assert sg_orig == sg_prim, f'Space group changed: {sg_orig} vs {sg_prim}'

# Check reduced atom count divides original evenly
assert len(atoms) % len(prim) == 0, 'Primitive atom count does not divide original evenly'

ase.io.write('structures/structure_000.cif', prim)
print('Wrote structures/structure_000.cif with', len(prim), 'atoms')
print('K-path:', kp['path'])
