"""
Generate 51 perturbed Cu structures (structure_000.cif .. structure_050.cif).

structure_000.cif is the unperturbed primitive fcc Cu cell (1 atom, a = 3.577 A).
The remaining 50 structures have:
  - random atomic displacements representative of thermal motion at 300 K
  - a small random lattice strain (|eps_ij| <= 0.005)

Temperature choice: 300 K.
fcc Cu (mp-30) is the stable phase of copper from 0 K up to its melting point
(1358 K), so 300 K is well within the stability range of this phase.

Displacement amplitudes:
  For a single-atom fcc cell the mean-square displacement U (in A^2) is
  estimated from the Debye model. Cu has Debye temperature ~343 K.
  At T = 300 K, U ~ (9 * hbar^2 * T) / (4 * pi^2 * M * theta_D^2) * f(T/theta_D)
  where f(x) ~ 1 for x << 1 (high-T limit) and ~ 3/4 for x ~ 1.
  More simply: the per-axis RMS displacement is sqrt(U) ~ 0.05 A at 300 K
  for Cu (literature values for fcc Cu at 300 K give U ~ 0.0025 A^2,
  so sqrt(U) ~ 0.05 A).

  We use RMS displacement = 0.05 A per axis for Cu.
  This is the per-axis RMS (sqrt(U)), NOT U itself.
  U = (0.05 A)^2 = 0.0025 A^2.

  Since there is only one species (Cu), all atoms get the same amplitude.
  If there were multiple species, lighter or more weakly bound atoms would
  get larger amplitudes (e.g. scale by 1/sqrt(M) or by 1/kappa).

Lattice strain:
  A symmetric 3x3 strain tensor with |eps_ij| <= 0.005 (0.5% max) is applied
  to the cell vectors. This is small enough to keep the structure close to
  equilibrium but large enough to sample different local environments.

Units:
  - Displacements: Angstrom (A)
  - Cell vectors: Angstrom (A)
  - Strain: dimensionless
"""

import numpy as np
import ase.io
from ase import Atoms
from ase.build import fcc111

# Read the base structure (already reduced to primitive cell)
base = ase.io.read('structures/structure_000.cif')
print(f"Base structure: {len(base)} atoms, symbols: {base.get_chemical_symbols()}")
print(f"Base cell (A):\n{base.cell}")

# Temperature and displacement parameters
T = 300.0  # K -- fcc Cu is stable from 0 K to 1358 K (melting point)

# Per-species RMS displacement (sqrt(U)) in Angstrom
# Cu at 300 K: U ~ 0.0025 A^2, so sqrt(U) ~ 0.05 A
# This is the per-axis RMS displacement, NOT the mean-square displacement U.
# U = (0.05)^2 = 0.0025 A^2
rms_displacement = {
    'Cu': 0.05,  # A -- per-axis RMS displacement (sqrt(U))
}

# Maximum strain magnitude (dimensionless)
max_strain = 0.005  # 0.5%

# Random seed for reproducibility
rng = np.random.default_rng(seed=42)

# Write structure_000.cif (unperturbed base)
ase.io.write('structures/structure_000.cif', base)
print("Wrote structures/structure_000.cif (unperturbed)")

# Generate 50 perturbed structures
for i in range(1, 51):
    # Start from a copy of the base structure
    atoms = base.copy()

    # Apply random atomic displacements
    # Each atom gets a random displacement drawn from a Gaussian distribution
    # with standard deviation = rms_displacement[species]
    # The displacement is in Cartesian coordinates (Angstrom)
    disp = np.zeros((len(atoms), 3))
    for j, sym in enumerate(atoms.get_chemical_symbols()):
        sigma = rms_displacement[sym]
        disp[j] = rng.normal(0.0, sigma, size=3)
    atoms.positions += disp

    # Apply small random lattice strain
    # Generate a symmetric strain tensor with |eps_ij| <= max_strain
    strain = rng.uniform(-max_strain, max_strain, size=(3, 3))
    strain = 0.5 * (strain + strain.T)  # make symmetric
    # Apply strain to the cell: new_cell = (I + strain) @ old_cell
    # Note: cell vectors are rows, so we apply strain as cell @ (I + strain).T
    new_cell = atoms.cell @ (np.eye(3) + strain).T
    atoms.cell = new_cell

    # INVARIANT 1: nat and ntyp must match the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 1, f"Expected 1 atom, got {nat}"
    assert ntyp == 1, f"Expected 1 type, got {ntyp}"

    # Write the structure
    fname = f'structures/structure_{i:03d}.cif'
    ase.io.write(fname, atoms)
    print(f"Wrote {fname} (nat={nat}, ntyp={ntyp})")

print("\nDone. Generated 51 structures (000-050).")
