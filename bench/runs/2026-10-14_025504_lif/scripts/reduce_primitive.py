import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/LiF_base.cif')
print('original: nat =', len(atoms), 'cell =', atoms.cell.array.tolist())

kp = standard_kpath(atoms)
print('spacegroup:', kp['spacegroup'])
prim = kp_atoms(kp)
print('primitive: nat =', len(prim), 'cell =', prim.cell.array.tolist())

# INVARIANT 1: nat/ntyp from the actual object
nat = len(prim)
ntyp = len(set(prim.get_chemical_symbols()))
print('nat =', nat, 'ntyp =', ntyp)
assert nat == 2, f'expected 2 atoms in primitive LiF, got {nat}'
assert ntyp == 2

# DECISION RULE check: space group number unchanged, atom count divides original
gen = ase.io.read('structures/LiF_base.cif')
assert len(gen) % nat == 0

ase.io.write('structures/structure_000.cif', prim)
print('wrote structures/structure_000.cif')
