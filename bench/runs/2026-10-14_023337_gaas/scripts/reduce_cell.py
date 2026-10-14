import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/gaas_base.cif')
print('original atoms:', len(atoms), 'symbols:', atoms.get_chemical_symbols())
print('original cell:', atoms.cell.array)

kp = standard_kpath(atoms)
print('spacegroup:', kp['spacegroup'])
prim = kp_atoms(kp)
print('primitive atoms:', len(prim), 'symbols:', prim.get_chemical_symbols())
print('primitive cell:', prim.cell.array)

# INVARIANT: space group number unchanged by reduction, and reduced count divides original
orig_sg = 216  # from generate_cif
prim_sg_num = int(kp['spacegroup'].split('#')[1].strip().strip(')'))
assert prim_sg_num == orig_sg, f'space group changed: {orig_sg} -> {prim_sg_num}'
assert len(atoms) % len(prim) == 0, 'reduced atom count does not divide original'

ase.io.write('structures/structure_000.cif', prim)
print('wrote structures/structure_000.cif with', len(prim), 'atoms')
