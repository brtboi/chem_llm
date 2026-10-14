import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/base_C.cif')
print('Original cell:', atoms.get_cell())
print('Original nat:', len(atoms))

kp = standard_kpath(atoms)
print('Space group:', kp['spacegroup'])

prim = kp_atoms(kp)
print('Primitive cell:', prim.get_cell())
print('Primitive nat:', len(prim))

# Save the primitive cell as the base structure for downstream use
ase.io.write('structures/base_C_prim.cif', prim)
print('Saved primitive cell to structures/base_C_prim.cif')
