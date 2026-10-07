import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms
from ase.spacegroup import get_spacegroup

base = ase.io.read('structures/CsPbBr3_base.cif')
print('Base cell atoms:', len(base))
print('Base cell:', base.cell.array)

kp = standard_kpath(base)
prim = kp_atoms(kp)
print('Primitive cell atoms:', len(prim))
print('Primitive cell:', prim.cell.array)

# Assert the reduction is sensible: primitive atom count divides original
assert len(base) % len(prim) == 0, 'primitive atom count must divide original'

# Save the primitive cell as the base structure for downstream steps
ase.io.write('structures/base_primitive.cif', prim)
print('Saved primitive cell to structures/base_primitive.cif')

# Verify space group is unchanged by the reduction
# Spacegroup object has .no (number) and .symbol (Hermann-Mauguin string)
sg_base = get_spacegroup(base, symprec=1e-4)
sg_prim = get_spacegroup(prim, symprec=1e-4)
print('Base space group:', sg_base.no, sg_base.symbol)
print('Primitive space group:', sg_prim.no, sg_prim.symbol)
assert sg_base.no == sg_prim.no, 'space group changed after reduction'
print('Space group preserved: OK')
