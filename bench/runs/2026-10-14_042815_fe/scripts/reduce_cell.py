import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/Fe_base.cif')
print('original cell:', atoms.get_cell())
print('original nat:', len(atoms))
print('original spacegroup (from kpath):', standard_kpath(atoms)['spacegroup'])

kp = standard_kpath(atoms)
prim = kp_atoms(kp)
print('primitive nat:', len(prim))
print('primitive spacegroup:', standard_kpath(prim)['spacegroup'])

# INVARIANT: space group number unchanged, atom count divides evenly
sg_orig = standard_kpath(atoms)['spacegroup']
sg_prim = standard_kpath(prim)['spacegroup']
assert sg_orig.split('#')[1].strip() == sg_prim.split('#')[1].strip(), f'space group changed: {sg_orig} vs {sg_prim}'
assert len(atoms) % len(prim) == 0, 'atom count does not divide evenly'

ase.io.write('structures/Fe_primitive.cif', prim)
print('wrote structures/Fe_primitive.cif')
print('kpath:', kp['path'])
