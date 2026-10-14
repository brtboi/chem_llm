#!/usr/bin/env python3
"""
Generate 51 perturbed wurtzite GaN structures (structure_000.cif .. structure_050.cif).

Structure 000: unperturbed primitive cell (from structures/gan_primitive.cif).
Structures 001-050: random atomic displacements (Einstein thermal model) + small random lattice strain.

Temperature choice: T = 300 K
  Wurtzite GaN is the stable phase at room temperature and up to ~1000 K.
  The hexagonal wurtzite phase is the ground-state phase of GaN at ambient
  conditions; the cubic zincblende phase only becomes competitive at very high
  pressures or temperatures. 300 K is well within the stability range.

Thermal displacement model (Einstein):
  For each species, the mean-square displacement U (in A^2) is estimated from
  the Einstein relation:
      U = 3 k_B T / (M * omega^2)
  where omega is an effective vibrational frequency.  We use a Debye-like
  estimate: omega ~ sqrt(k / M) with k ~ 100 N/m (typical for covalent bonds).
  The per-axis RMS displacement is sqrt(U) in A.

  Species amplitudes (RMS displacement per axis, in A):
    N  (M = 14.007 u): lighter atom, moves more.  U_N ~ 0.005 A^2 -> RMS ~ 0.071 A
    Ga (M = 69.723 u): heavier atom, moves less.  U_Ga ~ 0.001 A^2 -> RMS ~ 0.032 A

  These are representative of thermal motion at 300 K for a covalent semiconductor.

Lattice strain:
  Small random strain tensor with components drawn uniformly from [-0.005, 0.005]
  (0.5% max), applied to the cell vectors. This represents small lattice distortions
  from thermal expansion / phonon modes.

Units:
  - U: mean-square displacement in A^2
  - RMS per-axis displacement: sqrt(U) in A
  - Strain: dimensionless (fractional change in cell vectors)
"""

import numpy as np
import ase.io
from ase import Atoms
import os

# --- Parameters ---
T = 300.0  # K

# RMS displacement per axis (in A) for each species at T = 300 K
# Derived from Einstein model: U = 3*k_B*T/(M*omega^2), RMS = sqrt(U)
# N:  U ~ 0.005 A^2 -> RMS ~ 0.071 A
# Ga: U ~ 0.001 A^2 -> RMS ~ 0.032 A
RMS_DISP = {
    'N': 0.071,   # A, per-axis RMS displacement
    'Ga': 0.032,  # A, per-axis RMS displacement
}

# Max strain per component (dimensionless)
MAX_STRAIN = 0.005  # 0.5%

N_STRUCTURES = 51  # 000 (unperturbed) + 001..050 (perturbed)

# --- Read base structure ---
base = ase.io.read('structures/gan_primitive.cif')
print(f"Base structure: {len(base)} atoms, symbols: {base.get_chemical_symbols()}")
print(f"Cell:\n{base.cell.array}")

# INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
nat = len(base)
typ_set = set(base.get_chemical_symbols())
typ_set = set(base.get_chemical_symbols())
ntyp = len(typ_set)
print(f"nat = {nat}, ntyp = {ntyp}, species: {sorted(typ_set)}")

# --- Generate structures ---
os.makedirs('structures', exist_ok=True)

rng = np.random.default_rng(42)  # fixed seed for reproducibility

for i in range(N_STRUCTURES):
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()

        # --- Random lattice strain ---
        # Apply a small random strain tensor to the cell vectors
        # strain_ij is dimensionless; new_cell = (I + strain) @ cell
        strain = rng.uniform(-MAX_STRAIN, MAX_STRAIN, size=(3, 3))
        # Make strain symmetric (no shear rotation component for simplicity)
        strain = 0.5 * (strain + strain.T)
        new_cell = (np.eye(3) + strain) @ atoms.cell.array
        atoms.cell = new_cell

        # --- Random atomic displacements (Einstein thermal model) ---
        # Each atom gets a Gaussian displacement with per-species RMS amplitude
        # Displacements are in Cartesian coordinates (A)
        disp = np.zeros((nat, 3))
        for j, sym in enumerate(atoms.get_chemical_symbols()):
            rms = RMS_DISP[sym]
            disp[j] = rng.normal(0.0, rms, size=3)
        atoms.positions += disp

    # --- INVARIANT 1: assert nat/ntyp for each structure ---
    assert len(atoms) == nat, f"Structure {i:03d}: nat changed from {nat} to {len(atoms)}"
    assert len(set(atoms.get_chemical_symbols())) == ntyp, \
        f"Structure {i:03d}: ntyp changed from {ntyp} to {len(set(atoms.get_chemical_symbols()))}"

    # Write CIF
    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)

    if i == 0:
        print(f"Wrote {out_path} (unperturbed)")
    else:
        print(f"Wrote {out_path} (perturbed, strain max={MAX_STRAIN*100:.1f}%, "
              f"RMS_N={RMS_DISP['N']:.3f} A, RMS_Ga={RMS_DISP['Ga']:.3f} A)")

print(f"\nDone. Generated {N_STRUCTURES} structures in structures/")
