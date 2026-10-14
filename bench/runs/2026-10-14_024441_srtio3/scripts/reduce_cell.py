import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/SrTiO3_base.cif')
n_before = len(atoms)
sg_before = standard_kpath(atoms)['spacegroup']
print('Before reduction:', n_before, 'atoms, space group', sg_before)

kp = standard_kpath(atoms)
prim = kp_atoms(kp)
n_after = len(prim)
sg_after = standard_kpath(prim)['spacegroup']
print('After reduction:', n_after, 'atoms, space group', sg_after)
print('k-path:', kp['path'])

# INVARIANT: space group unchanged, atom count divides evenly
assert sg_before.split('#')[1].strip() == sg_after.split('#')[1].strip(), \
    f'space group changed: {sg_before} vs {sg_after}'
assert n_before % n_after == 0, f'{n_before} not divisible by {n_after}'

ase.io.write('structures/structure_000.cif', prim)
print('Wrote structures/structure_000.cif')
print('nat =', len(prim), 'ntyp =', len(set(prim.get_chemical_symbols())))
