import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/base_Si.cif')
print(f"Original cell: {len(atoms)} atoms, cell={atoms.cell}")

kp = standard_kpath(atoms)
print(f"Space group: {kp['spacegroup']}")
prim = kp_atoms(kp)
print(f"Primitive cell: {len(prim)} atoms, cell={prim.cell}")

# Assert space group unchanged
kp2 = standard_kpath(prim)
assert kp2['spacegroup'] == kp['spacegroup'], f"Space group changed: {kp['spacegroup']} -> {kp2['spacegroup']}"

# Assert reduced atom count divides original
assert len(atoms) % len(prim) == 0, f"{len(prim)} does not divide {len(atoms)}"

ase.io.write('structures/prim_Si.cif', prim)
print(f"Saved primitive cell to structures/prim_Si.cif")
print(f"Atom count: {len(atoms)} -> {len(prim)}")
