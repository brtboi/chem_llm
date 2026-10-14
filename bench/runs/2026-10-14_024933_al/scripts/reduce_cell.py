import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/Al_base.cif')
print(f"Original cell: {len(atoms)} atoms, cell={atoms.cell[:].tolist()}")

kp = standard_kpath(atoms)
print(f"Space group: {kp['spacegroup']}")
prim = kp_atoms(kp)
print(f"Primitive cell: {len(prim)} atoms, cell={prim.cell[:].tolist()}")

# Assert space group unchanged: parse the number from 'Fm-3m (#225)'
sg_num = int(kp['spacegroup'].split('#')[-1].strip().rstrip(')'))
assert sg_num == 225, f"Space group changed: {kp['spacegroup']}"
# Assert reduced atom count divides original
assert len(atoms) % len(prim) == 0, f"{len(prim)} does not divide {len(atoms)}"

ase.io.write('structures/Al_primitive.cif', prim)
print("Saved primitive cell to structures/Al_primitive.cif")
