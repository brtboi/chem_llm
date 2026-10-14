import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/base_Bi2Se3.cif')
print('Original cell:', atoms.get_cell())
print('Original nat:', len(atoms))
print('Original ntyp:', len(set(atoms.get_chemical_symbols())))
print('Original symbols:', atoms.get_chemical_symbols())

kp = standard_kpath(atoms)
print('Space group:', kp['spacegroup'])
prim = kp_atoms(kp)
print('Primitive cell:', prim.get_cell())
print('Primitive nat:', len(prim))
print('Primitive ntyp:', len(set(prim.get_chemical_symbols())))
print('Primitive symbols:', prim.get_chemical_symbols())

# INVARIANT 1: nat/ntyp computed from actual Atoms object
assert len(prim) > 0
assert len(set(prim.get_chemical_symbols())) == 2  # Bi and Se

# Save the primitive cell as structure_000.cif (unperturbed base)
ase.io.write('structures/structure_000.cif', prim)
print('Saved structures/structure_000.cif')
print('Atom count before reduction:', len(atoms))
print('Atom count after reduction:', len(prim))
