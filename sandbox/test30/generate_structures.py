import numpy as np
from ase.io import read, write
from ase import Atoms

# Read the primitive cell (6 atoms: 2 Ti + 4 O) as the base structure
base = read('structures/TiO2_primitive.cif')
n_base = len(base)
ntyp_base = len(set(base.get_chemical_symbols()))
print(f"Base structure: {n_base} atoms, {ntyp_base} element types")

# Displacement magnitude justification:
# At room temperature (300 K), typical thermal displacement for TiO2 is ~0.05 A.
# This is based on Debye-Waller factors: U ~ 0.001-0.01 A^2, so sqrt(3*U) ~ 0.05-0.17 A.
# We use 0.05 A as a conservative estimate that captures thermal motion without
# causing unphysical bond breaking or large structural distortions.
DISPLACEMENT_MAGNITUDE = 0.05  # Angstroms

# Random strain magnitude: ~1% is small enough to preserve tetragonal character
# while sampling lattice distortions that might occur in real materials.
STRAIN_MAGNITUDE = 0.01  # 1%

np.random.seed(42)  # For reproducibility

# Structure 000: unperturbed base
write('structures/structure_000.cif', base)
print(f"Wrote structures/structure_000.cif (unperturbed base)")

# Structures 001-050: perturbed copies
for i in range(1, 51):
    atoms = base.copy()
    
    # Apply random atomic displacements (Gaussian, zero mean, sigma = DISPLACEMENT_MAGNITUDE)
    displacements = np.random.normal(0, DISPLACEMENT_MAGNITUDE, size=(len(atoms), 3))
    atoms.positions += displacements
    
    # Apply small random lattice strain
    # Generate a random strain tensor (symmetric, traceless for volume preservation)
    strain = np.random.normal(0, STRAIN_MAGNITUDE, size=(3, 3))
    strain = (strain + strain.T) / 2  # Make symmetric
    strain -= np.trace(strain) / 3 * np.eye(3)  # Make traceless (volume preserving)
    
    # Apply strain to the cell
    new_cell = (np.eye(3) + strain) @ atoms.cell[:]
    atoms.cell = new_cell
    
    # INVARIANT 1: Assert nat/ntyp match the actual structure
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == n_base, f"Structure {i:03d}: nat changed from {n_base} to {nat}"
    assert ntyp == ntyp_base, f"Structure {i:03d}: ntyp changed from {ntyp_base} to {ntyp}"
    
    filename = f'structures/structure_{i:03d}.cif'
    write(filename, atoms)
    print(f"Wrote {filename} (nat={nat}, ntyp={ntyp})")

print(f"\nGenerated 51 structures total (000-050)")
print(f"All structures have nat={n_base}, ntyp={ntyp_base}")
