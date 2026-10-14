import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/LiF_base.cif')
n_before = len(atoms)
print(f"Original cell: {n_before} atoms, cell={atoms.cell[:].tolist()}")

kp = standard_kpath(atoms)
print(f"Space group: {kp['spacegroup']}")
prim = kp_atoms(kp)
n_after = len(prim)
print(f"Primitive cell: {n_after} atoms, cell={prim.cell[:].tolist()}")

# INVARIANT: space group unchanged, atom count divides evenly
sg_num = int(kp['spacegroup'].split('#')[-1].strip().rstrip(')'))
assert n_before % n_after == 0, f"{n_before} not divisible by {n_after}"
print(f"Space group number: {sg_num}")
print(f"Reduction factor: {n_before // n_after}")

# Save the primitive cell for downstream use
ase.io.write('structures/LiF_primitive.cif', prim)
print("Saved structures/LiF_primitive.cif")
