#!/usr/bin/env python3
"""
Generate 51 rutile TiO2 structures: structure_000.cif (unperturbed base)
and structure_001.cif through structure_050.cif (perturbed copies).

Perturbation scheme:
  - Random atomic displacements: Gaussian, sigma = 0.05 Angstrom per Cartesian
    component. Justification: at room temperature (300 K), the mean-square
    displacement of atoms in a typical ionic crystal is on the order of
    0.01-0.05 Angstrom (Debye-Waller factors for TiO2 give U ~ 0.001-0.01 A^2,
    so sigma ~ 0.03-0.1 A). 0.05 A is a reasonable upper-bound estimate that
    captures thermal motion without breaking bonds.
  - Random lattice strain: uniform random strain tensor with components in
    [-0.01, 0.01] (i.e., up to 1% strain). Justification: small elastic
    deformations that do not change the crystal structure but sample the
    local energy landscape.

INVARIANT 1: nat and ntyp are computed from the actual ASE Atoms object,
never hardcoded.
"""

import numpy as np
from ase.io import read, write
from ase import Atoms

# Read the primitive base structure (6 atoms, space group 136)
base = read('structures/rutile_TiO2_primitive.cif')
n_base = len(base)
ntyp_base = len(set(base.get_chemical_symbols()))
print(f"Base structure: {n_base} atoms, {ntyp_base} element types")
print(f"Base cell:\n{base.cell}")

# Set a fixed random seed for reproducibility
rng = np.random.default_rng(seed=42)

# Structure 000: unperturbed base
write('structures/structure_000.cif', base)
print("Wrote structures/structure_000.cif (unperturbed)")

# Structures 001-050: perturbed copies
for i in range(1, 51):
    atoms = base.copy()

    # Random atomic displacements: Gaussian, sigma = 0.05 A per component
    # This is a chemically reasonable estimate of room-temperature thermal
    # motion for TiO2 (Debye-Waller analysis gives sigma ~ 0.03-0.1 A).
    displacements = rng.normal(loc=0.0, scale=0.05, size=atoms.positions.shape)
    atoms.positions += displacements

    # Small random lattice strain: uniform in [-0.01, 0.01] per component
    # This samples small elastic deformations (up to 1%) without changing
    # the crystal structure.
    strain = rng.uniform(low=-0.01, high=0.01, size=(3, 3))
    # Apply strain to the cell: new_cell = (I + strain) @ old_cell
    # Note: strain is applied in the Cartesian basis
    new_cell = (np.eye(3) + strain) @ atoms.cell[:]
    atoms.cell = new_cell

    # INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == n_base, f"Structure {i:03d}: nat changed from {n_base} to {nat}"
    assert ntyp == ntyp_base, f"Structure {i:03d}: ntyp changed from {ntyp_base} to {ntyp}"

    # Verify the structure is still physically reasonable:
    # - No atoms should be too close (minimum distance > 0.3 A)
    # - Cell volume should be positive
    distances = atoms.get_all_distances()
    # Exclude self-distances (diagonal)
    np.fill_diagonal(distances, np.inf)
    min_dist = distances.min()
    assert min_dist > 0.3, f"Structure {i:03d}: minimum distance {min_dist:.3f} A is too small"

    vol = atoms.get_volume()
    assert vol > 0, f"Structure {i:03d}: cell volume {vol:.3f} A^3 is not positive"

    out_path = f'structures/structure_{i:03d}.cif'
    write(out_path, atoms)
    print(f"Wrote {out_path} (nat={nat}, ntyp={ntyp}, min_dist={min_dist:.3f} A, vol={vol:.2f} A^3)")

print("\nDone: 51 structures written (000-050)")
