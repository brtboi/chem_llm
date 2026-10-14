import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

# Read the base CIF (conventional cell from Materials Project)
base = ase.io.read('structures/zno_base.cif')
print(f"Base cell: {len(base)} atoms, cell = {base.cell}")

# The primitive cell for the k-path is obtained via chem_llm.kpath
kp = standard_kpath(base)
print(f"Space group: {kp['spacegroup']}")
prim = kp_atoms(kp)
print(f"Primitive cell: {len(prim)} atoms")

# INVARIANT: space group number must be unchanged by the reduction
# and the reduced atom count must divide the original evenly
base_sg = standard_kpath(base)['spacegroup']
prim_sg = standard_kpath(prim)['spacegroup']
assert base_sg == prim_sg, f"Space group changed: {base_sg} -> {prim_sg}"
assert len(base) % len(prim) == 0, f"{len(base)} not divisible by {len(prim)}"

# Save the primitive cell as the base for generate_structures.py
ase.io.write('structures/zno_primitive.cif', prim)
print(f"Saved primitive cell to structures/zno_primitive.cif")
print(f"Atom count: {len(base)} -> {len(prim)}")
