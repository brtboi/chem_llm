import numpy as np
from ase.io import read, write
from chem_llm.kpath import standard_kpath, kp_atoms
from spglib import get_symmetry_dataset

# Read the base CIF (mp-48, graphite, P6_3/mmc #194)
base = read('structures/base_C.cif')
print('Base cell: nat =', len(base), 'cell =\n', base.cell)

# Reduce to the primitive cell using seekpath (same code as get_kpath).
# The primitive cell is what the k-path refers to and what we run DFT on.
kp = standard_kpath(base)
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim), 'cell =\n', prim.cell)

# DECISION RULE checks: atom count divides evenly; space group unchanged.
assert len(base) % len(prim) == 0, 'primitive atom count must divide original'

# spglib expects (lattice, scaled_positions, numbers), not an ASE Atoms object.
sg_base = get_symmetry_dataset((base.cell.array, base.get_scaled_positions(), base.get_atomic_numbers()), symprec=1e-5)
sg_prim = get_symmetry_dataset((prim.cell.array, prim.get_scaled_positions(), prim.get_atomic_numbers()), symprec=1e-5)
print('Space group base:', sg_base['international'], 'number', sg_base['number'])
print('Space group prim:', sg_prim['international'], 'number', sg_prim['number'])
assert sg_base['number'] == sg_prim['number'] == 194, 'space group changed by reduction'

# Save the primitive cell as the base structure for the rest of the workflow
write('structures/base_C_prim.cif', prim)
print('Saved structures/base_C_prim.cif with', len(prim), 'atoms')
print('k-path:', kp['path'])
