import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/base.cif')
print(f"Original cell: {len(atoms)} atoms, {len(set(atoms.get_chemical_symbols()))} species")
print(f"Cell: {atoms.cell[:].tolist()}")

kp = standard_kpath(atoms)
print(f"Space group from kpath: {kp['spacegroup']}")

prim = kp_atoms(kp)
print(f"Primitive cell: {len(prim)} atoms, {len(set(prim.get_chemical_symbols()))} species")
print(f"Primitive cell vectors: {prim.cell[:].tolist()}")

# INVARIANT: space group unchanged by reduction
assert '221' in kp['spacegroup'], f"Space group changed: {kp['spacegroup']}"

# INVARIANT: reduced atom count divides original evenly
assert len(atoms) % len(prim) == 0, f"{len(prim)} does not divide {len(atoms)}"

print(f"\nReduction factor: {len(atoms)}/{len(prim)} = {len(atoms)//len(prim)}")
print(f"Species in primitive: {prim.get_chemical_symbols()}")

# Save the primitive cell as the base for downstream use
ase.io.write('structures/base_primitive.cif', prim)
print("\nSaved primitive cell to structures/base_primitive.cif")
