import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/GaAs_base.cif')
print('Original cell:', atoms.get_cell())
print('Original atom count:', len(atoms))
print('Original symbols:', atoms.get_chemical_symbols())

kp = standard_kpath(atoms)
print('Space group:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell:', prim.get_cell())
print('Primitive atom count:', len(prim))

# INVARIANT: space group number unchanged, reduced count divides original
def sg_num(s):
    return int(s.split('#')[-1].strip().strip('()'))

assert sg_num(kp['spacegroup']) == 216, f"Space group changed: {kp['spacegroup']}"
assert len(atoms) % len(prim) == 0, "Reduced atom count does not divide original"

ase.io.write('structures/GaAs_primitive.cif', prim)
print('Saved primitive cell to structures/GaAs_primitive.cif')
