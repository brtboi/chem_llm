import numpy as np
import seekpath
import spglib
from ase.io import read, write
from ase import Atoms

# Read base CIF
atoms = read('structures/CsPbBr3_base.cif')
print(f"Base cell: {len(atoms)} atoms, cell:\n{atoms.cell}")
print(f"Space group from CIF metadata: {atoms.info.get('space_group', 'unknown')}")

# Build cell_tuple for seekpath
cell_tuple = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())

# Get symmetry dataset with symprec=0.01 (matching pymatgen default, per INVARIANT 3)
dataset = spglib.get_symmetry_dataset(cell_tuple, symprec=0.01)
print(f"spglib space group number: {dataset.number}")
print(f"spglib space group symbol: {dataset.international}")

# Get k-path and primitive cell from seekpath
res = seekpath.get_path(cell_tuple, symprec=0.01)
print(f"seekpath space group number: {res['spacegroup_number']}")
print(f"seekpath path: {res['path']}")
print(f"seekpath point_coords: {res['point_coords']}")

# INVARIANT 3 check: seekpath and spglib must agree
assert res['spacegroup_number'] == dataset.number, \
    f"Space group mismatch: seekpath={res['spacegroup_number']}, spglib={dataset.number}"

# Build primitive cell from seekpath's own reduced cell
prim = Atoms(
    cell=res['primitive_lattice'],
    scaled_positions=res['primitive_positions'],
    numbers=res['primitive_types'],
    pbc=True
)
print(f"\nPrimitive cell: {len(prim)} atoms")
print(f"Primitive cell vectors:\n{prim.cell}")
print(f"Primitive atomic numbers: {prim.get_atomic_numbers()}")
print(f"Primitive chemical symbols: {prim.get_chemical_symbols()}")

# DECISION RULE check: reduced atom count divides original evenly
assert len(atoms) % len(prim) == 0, \
    f"Reduced atom count {len(prim)} does not divide original {len(atoms)}"

# Verify space group unchanged by reduction
prim_cell_tuple = (prim.cell[:], prim.get_scaled_positions(), prim.get_atomic_numbers())
prim_dataset = spglib.get_symmetry_dataset(prim_cell_tuple, symprec=0.01)
print(f"\nPrimitive cell spglib space group: {prim_dataset.number} ({prim_dataset.international})")
assert prim_dataset.number == dataset.number, \
    f"Space group changed after reduction: {dataset.number} -> {prim_dataset.number}"

# Save primitive cell
write('structures/CsPbBr3_primitive.cif', prim)
print("\nSaved primitive cell to structures/CsPbBr3_primitive.cif")

# Record nat/ntyp for downstream use
nat = len(prim)
typ = len(set(prim.get_chemical_symbols()))
print(f"\nnat = {nat}, ntyp = {typ}")
print(f"Elements: {sorted(set(prim.get_chemical_symbols()))}")

# Save k-path info for setup_jobs.py
# point_coords is a dict mapping label -> list of 3 floats (already JSON-serializable)
# primitive_lattice/positions/types are numpy arrays -> convert to lists
import json
path_info = {
    'path': res['path'],
    'point_coords': {k: list(v) for k, v in res['point_coords'].items()},
    'primitive_lattice': np.array(res['primitive_lattice']).tolist(),
    'primitive_positions': np.array(res['primitive_positions']).tolist(),
    'primitive_types': np.array(res['primitive_types']).tolist(),
    'spacegroup_number': res['spacegroup_number'],
}
with open('structures/kpath_info.json', 'w') as f:
    json.dump(path_info, f, indent=2)
print("Saved k-path info to structures/kpath_info.json")
