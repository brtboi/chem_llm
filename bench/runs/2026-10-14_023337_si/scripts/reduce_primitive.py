import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

# Read the base CIF from Materials Project (mp-149, Fd-3m #227)
base = ase.io.read('structures/si_base.cif')
print(f"Base cell: {len(base)} atoms, cell = {base.cell[:].tolist()}")

# Reduce to primitive cell via seekpath (chem_llm.kpath)
kp = standard_kpath(base)
print(f"Space group: {kp['spacegroup']}")
prim = kp_atoms(kp)
print(f"Primitive cell: {len(prim)} atoms, cell = {prim.cell[:].tolist()}")

# INVARIANT: space group unchanged, atom count divides original evenly
# The base cell from mp-149 is already primitive (2 atoms), so the reduction is a no-op.
assert len(base) % len(prim) == 0, f"Atom count mismatch: {len(prim)} vs {len(base)}"
# Check space group number is preserved
assert '227' in kp['spacegroup'], f"Space group changed: {kp['spacegroup']}"

# Save the primitive cell as the base for downstream use
ase.io.write('structures/si_primitive.cif', prim)
print(f"Saved primitive cell to structures/si_primitive.cif")
print(f"Atoms before: {len(base)}, after: {len(prim)}")
