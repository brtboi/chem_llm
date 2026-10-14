import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/TiO2_base.cif')
print(f"Original cell: {len(atoms)} atoms, cell={atoms.cell}")

kp = standard_kpath(atoms)
print(f"Space group: {kp['spacegroup']}")
prim = kp_atoms(kp)
print(f"Primitive cell: {len(prim)} atoms, cell={prim.cell}")

# INVARIANT: space group unchanged, atom count divides original
assert '136' in kp['spacegroup'], f"Space group changed: {kp['spacegroup']}"
assert len(atoms) % len(prim) == 0, "Primitive cell atom count does not divide original"

ase.io.write('structures/structure_000.cif', prim)
print(f"Wrote structures/structure_000.cif with {len(prim)} atoms")
print(f"nat={len(prim)}, ntyp={len(set(prim.get_chemical_symbols()))}")
