import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/gan_base.cif')
print(f"Original: {len(atoms)} atoms, cell={atoms.cell[:].tolist()}")
print(f"Space group from CIF: {atoms.info.get('space_group', 'unknown')}")

kp = standard_kpath(atoms)
print(f"Space group from seekpath: {kp['spacegroup']}")

prim = kp_atoms(kp)
print(f"Primitive: {len(prim)} atoms, cell={prim.cell[:].tolist()}")

# Assert space group number unchanged
assert '186' in kp['spacegroup'], f"Space group changed: {kp['spacegroup']}"
# Assert reduced atom count divides original
assert len(atoms) % len(prim) == 0, f"{len(prim)} does not divide {len(atoms)}"

ase.io.write('structures/structure_000_base.cif', prim)
print("Saved primitive cell to structures/structure_000_base.cif")
