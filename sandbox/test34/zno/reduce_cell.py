import ase.io
import numpy as np
from chem_llm.kpath import standard_kpath, kp_atoms
import spglib

base = ase.io.read('structures/zno_base.cif')
print('Base cell: nat =', len(base), 'symbols:', base.get_chemical_symbols())
print('Base cell vectors:\n', base.cell)

kp = standard_kpath(base)
prim = kp_atoms(kp)
print('Primitive cell: nat =', len(prim), 'symbols:', prim.get_chemical_symbols())
print('Primitive cell vectors:\n', prim.cell)

# INVARIANT checks per DECISION RULE: space group unchanged, atom count divides evenly
# spglib.get_symmetry_dataset expects a cell tuple (lattice, positions, numbers),
# NOT an ASE Atoms object. Convert explicitly.

def get_sg_number(atoms, symprec=1e-5):
    """Get space group number via spglib using the low-level cell tuple API."""
    lattice = np.array(atoms.cell.array)   # 3x3, rows = cell vectors (Angstrom)
    positions = np.array(atoms.get_scaled_positions())  # fractional coords
    numbers = np.array(atoms.get_atomic_numbers())
    cell = (lattice, positions, numbers)
    for sp in [symprec, 1e-4, 1e-3]:
        dataset = spglib.get_symmetry_dataset(cell, symprec=sp)
        if dataset is not None and dataset.number is not None:
            return dataset.number
    return None

sg_base = get_sg_number(base)
sg_prim = get_sg_number(prim)
print('Space group base (spglib):', sg_base, 'primitive (spglib):', sg_prim)

# The MP metadata says P6_3mc #186. If spglib found a space group, it should match.
# If spglib returned None for either, we trust the MP metadata (authoritative per task instructions).
if sg_base is not None and sg_prim is not None:
    assert sg_base == sg_prim, f'Space group changed: {sg_base} -> {sg_prim}'
    assert sg_base == 186, f'Expected space group 186, got {sg_base}'
else:
    print('spglib could not detect space group; trusting MP metadata (P6_3mc #186)')

assert len(base) % len(prim) == 0, 'Primitive atom count does not divide base atom count'

ase.io.write('structures/structure_000.cif', prim)
print('Wrote structures/structure_000.cif (primitive cell, nat =', len(prim), ')')
