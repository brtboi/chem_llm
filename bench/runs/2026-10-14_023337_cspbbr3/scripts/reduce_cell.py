import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/base_CsPbBr3.cif')
print('Original cell: nat =', len(atoms))
print('Space group from CIF metadata:', atoms.info.get('space_group', 'unknown'))

kp = standard_kpath(atoms)
print('Space group from seekpath:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim))

# INVARIANT: reduced atom count divides original evenly
assert len(atoms) % len(prim) == 0, f"{len(prim)} does not divide {len(atoms)}"

# Space group number should be unchanged
sg_orig = kp['spacegroup']
print('Space group (should be Pm-3m #221):', sg_orig)
assert '221' in sg_orig, f"Space group changed: {sg_orig}"

ase.io.write('structures/structure_000.cif', prim)
print('Wrote structures/structure_000.cif with', len(prim), 'atoms')
print('Cell vectors:', prim.cell.array.tolist())
print('Atomic positions (fractional):')
for i, a in enumerate(prim):
    print(f'  {a.symbol} {a.position.tolist()}')
