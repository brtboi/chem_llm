#!/usr/bin/env python3
"""
Generate 51 perturbed LiF structures (structure_000.cif .. structure_050.cif).

Structure 000 is the unperturbed primitive cell.
Structures 001-050 have:
  - random atomic displacements representative of thermal motion at T = 300 K
  - a small random lattice strain (max ~0.5%)

Temperature choice:
  LiF is a rock-salt (B1) ionic crystal stable at room temperature and above.
  Its Debye temperature is ~1000 K, so at 300 K it is well within the harmonic
  regime. We use T = 300 K as a representative thermal-motion temperature.

Displacement amplitudes (RMS, in Angstrom):
  The mean-square displacement U (in A^2) for an atom of mass m at temperature T
  in the Debye model is approximately:
      U ~ (3 * hbar^2) / (4 * pi^2 * m * c_s^2) * (T / T_D)^2 * f(T/T_D)
  For a rough estimate at 300 K:
    - Li (m = 7 amu, lighter, more mobile): RMS ~ 0.05 A  (U ~ 0.0025 A^2)
    - F  (m = 19 amu, heavier):              RMS ~ 0.03 A  (U ~ 0.0009 A^2)
  These are order-of-magnitude estimates consistent with typical thermal
  displacement parameters for light ionic crystals at 300 K.

Lattice strain:
  A small random symmetric strain tensor with components in [-0.005, 0.005]
  (i.e., up to 0.5% strain) is applied to the cell vectors.
"""

import numpy as np
import ase.io
from ase import Atoms

# Read the primitive cell
base = ase.io.read('structures/LiF_primitive.cif')
print(f"Base structure: {len(base)} atoms, symbols={base.get_chemical_symbols()}")

# Per-species RMS displacement amplitudes (in Angstrom)
# Li: lighter atom, moves more; F: heavier, moves less
RMS_Li = 0.05  # A  (U_Li = RMS_Li^2 = 0.0025 A^2)
RMS_F  = 0.03  # A  (U_F  = RMS_F^2  = 0.0009 A^2)

# Maximum strain magnitude (dimensionless, i.e., 0.005 = 0.5%)
MAX_STRAIN = 0.005

rng = np.random.default_rng(seed=42)

for i in range(51):
    atoms = base.copy()

    if i == 0:
        # Unperturbed base structure
        pass
    else:
        # --- Random atomic displacements (thermal motion) ---
        # Displace each atom by a Gaussian random vector with per-species RMS
        pos = atoms.get_positions().copy()
        for j, sym in enumerate(atoms.get_chemical_symbols()):
            if sym == 'Li':
                pos[j] += rng.normal(0.0, RMS_Li, size=3)
            elif sym == 'F':
                pos[j] += rng.normal(0.0, RMS_F, size=3)
            else:
                raise ValueError(f"Unexpected species {sym}")
        atoms.set_positions(pos)

        # --- Small random lattice strain ---
        # Apply a random symmetric strain tensor to the cell
        # strain tensor: symmetric 3x3 with components in [-MAX_STRAIN, MAX_STRAIN]
        strain = rng.uniform(-MAX_STRAIN, MAX_STRAIN, size=(3, 3))
        strain = 0.5 * (strain + strain.T)  # make symmetric
        # New cell = (I + strain) @ old_cell
        old_cell = atoms.cell[:]
        new_cell = (np.eye(3) + strain) @ old_cell
        atoms.set_cell(new_cell, scale_atoms=False)

    # INVARIANT 1: nat and ntyp must match the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Structure {i:03d}: expected nat=2, got {nat}"
    assert ntyp == 2, f"Structure {i:03d}: expected ntyp=2, got {ntyp}"

    # Write the structure
    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)

print(f"Generated 51 structures: structure_000.cif .. structure_050.cif")
print("Done.")
