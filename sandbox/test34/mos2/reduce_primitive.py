import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

# Read the base MoS2 CIF (2H, P6_3/mmc #194, mp-2815)
base = ase.io.read('structures/MoS2_base.cif')
n_base = len(base)
print(f'Base cell: {n_base} atoms, cell = {base.cell[:]}')

# Reduce to the primitive cell that the k-path refers to (seekpath).
# kp_atoms returns the primitive cell; the k-path's fractional k-points
# refer to ITS reciprocal basis, so this is the cell all downstream
# QE inputs must use (INVARIANT 3 / unit-cell-choice rule).
kp = standard_kpath(base)
prim = kp_atoms(kp)
n_prim = len(prim)
print(f'Primitive cell: {n_prim} atoms, cell = {prim.cell[:]}')

# standard_kpath returns a dict; print its keys to learn the schema.
print(f'kp keys: {list(kp.keys())}')
for k, v in kp.items():
    if not isinstance(v, (list, tuple)) or len(str(v)) < 200:
        print(f'  {k}: {v}')

# CHECK IN CODE per the unit-cell-choice rule: the reduced atom count
# must divide the original evenly.
assert n_base % n_prim == 0, f'{n_base} not divisible by {n_prim}'

# Save the primitive cell as the base structure for the workflow.
ase.io.write('structures/MoS2_primitive.cif', prim)
print(f'Saved structures/MoS2_primitive.cif ({n_prim} atoms)')
print(f'Before reduction: {n_base} atoms, after: {n_prim} atoms')
