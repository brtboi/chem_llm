import numpy as np
from ase.io import read, write
from ase import Atoms

# Read the primitive base structure (6 atoms, Ti2O4, P4_2/mnm #136)
base = read('structures/TiO2_primitive.cif')
n_base = len(base)
ntyp_base = len(set(base.get_chemical_symbols()))
print(f"Base structure: {n_base} atoms, {ntyp_base} species")

# Displacement magnitude justification:
# At room temperature (300 K), the typical atomic displacement in a solid is on the
# order of 0.01-0.05 Angstrom (from Debye-Waller factors). We use 0.05 A as a
# chemically reasonable upper bound for random thermal displacements that do not
# break bonds or cause unphysical overlaps in TiO2 (Ti-O bond ~1.95 A, O-O ~2.9 A).
# Lattice strain: 1% random strain is small enough to preserve the tetragonal
# character while sampling a range of lattice distortions.
DISPLACEMENT = 0.05  # Angstrom
STRAIN = 0.01  # 1% random strain

rng = np.random.default_rng(42)  # fixed seed for reproducibility

# Structure 000: unperturbed base
write('structures/structure_000.cif', base)
print(f"Wrote structures/structure_000.cif (unperturbed base)")

for i in range(1, 51):
    atoms = base.copy()
    
    # Random atomic displacements (Cartesian, in Angstrom)
    displacements = rng.normal(0.0, DISPLACEMENT, size=(len(atoms), 3))
    atoms.positions += displacements
    
    # Small random lattice strain: apply a random symmetric strain tensor
    # strain = random symmetric matrix with elements ~ STRAIN
    strain_tensor = rng.normal(0.0, STRAIN, size=(3, 3))
    strain_tensor = 0.5 * (strain_tensor + strain_tensor.T)  # make symmetric
    # New cell = (I + strain) @ old_cell
    new_cell = (np.eye(3) + strain_tensor) @ atoms.cell.array
    atoms.cell = new_cell
    
    # INVARIANT 1: assert nat/ntyp are consistent with the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == n_base, f"Structure {i:03d}: nat changed from {n_base} to {nat}"
    assert ntyp == ntyp_base, f"Structure {i:03d}: ntyp changed from {ntyp_base} to {ntyp}"
    
    fname = f'structures/structure_{i:03d}.cif'
    write(fname, atoms)
    print(f"Wrote {fname} (nat={nat}, ntyp={ntyp})")

print(f"\nDone: wrote 51 structures (structure_000.cif through structure_050.cif)")
