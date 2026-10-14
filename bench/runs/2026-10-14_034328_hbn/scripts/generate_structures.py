#!/usr/bin/env python3
"""
Generate 51 perturbed hBN structures (structure_000.cif .. structure_050.cif).

structure_000.cif is the unperturbed primitive cell.
structure_001..050 are perturbed copies with:
  - random atomic displacements representative of thermal motion at 300 K
  - small random lattice strain (<= 0.5%)

Temperature choice: 300 K (room temperature).
hBN is the thermodynamically stable phase of BN at ambient conditions.
The hexagonal (wurtzite-like) structure is stable up to very high temperatures;
the cubic (zincblende) phase only becomes competitive above ~1000 K under pressure.
300 K is therefore a physically meaningful temperature at which this phase is stable.

Displacement amplitudes (RMS, in Angstrom):
  The per-axis RMS displacement is sqrt(U) where U is the mean-square displacement in A^2.
  We use a simple Einstein-oscillator estimate:
      U = k_B * T / (M * omega^2)
  where omega is an effective vibrational frequency.
  For hBN, typical phonon frequencies are ~700 cm^-1 for B and ~900 cm^-1 for N.
  Converting: omega = 2*pi*c*nu  (nu in cm^-1, c = 2.998e10 cm/s)

  For B (M = 10.81 amu = 1.799e-26 kg):
      omega_B = 2*pi*2.998e10*700 = 1.322e14 rad/s
      U_B = 1.381e-23 * 300 / (1.799e-26 * (1.322e14)^2)
           = 4.143e-21 / 3.138e-2 = 1.320e-19 m^2 = 1.320e-5 A^2
      RMS_B = sqrt(1.320e-5) = 3.63e-3 A

  For N (M = 14.01 amu = 2.325e-26 kg):
      omega_N = 2*pi*2.998e10*900 = 1.698e14 rad/s
      U_N = 1.381e-23 * 300 / (2.325e-26 * (1.698e14)^2)
           = 4.143e-21 / 6.737e-2 = 6.150e-20 m^2 = 6.150e-6 A^2
      RMS_N = sqrt(6.150e-6) = 2.48e-3 A

  B is lighter and has a lower frequency, so it moves more (larger RMS).
  N is heavier and stiffer, so it moves less.

  We use these RMS values directly as the standard deviation of a Gaussian
  displacement distribution for each Cartesian component.

Lattice strain: uniform random strain tensor with components in [-0.005, 0.005]
(0.5% max), applied to the cell vectors.
"""

import numpy as np
import ase.io
from ase import Atoms

# --- Read the base (primitive) structure ---
base = ase.io.read('structures/BN_primitive.cif')
print(f"Base structure: {len(base)} atoms, symbols={base.get_chemical_symbols()}")

# --- Displacement parameters (RMS in Angstrom, per Cartesian axis) ---
# Derived from Einstein-oscillator model at T = 300 K (see docstring for details).
# These are RMS displacements (sqrt(U)), NOT mean-square displacements (U).
T = 300.0  # K
RMS_B = 3.63e-3  # Angstrom  (sqrt(U_B), U_B = 1.320e-5 A^2)
RMS_N = 2.48e-3  # Angstrom  (sqrt(U_N), U_N = 6.150e-6 A^2)

# Map element -> RMS displacement (Angstrom)
RMS_MAP = {'B': RMS_B, 'N': RMS_N}

# --- Lattice strain parameters ---
MAX_STRAIN = 0.005  # 0.5% max strain per component

# --- Random seed for reproducibility ---
np.random.seed(42)

for idx in range(51):
    if idx == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()

        # --- Apply random lattice strain ---
        # Strain tensor: small random values in [-MAX_STRAIN, MAX_STRAIN]
        strain = np.random.uniform(-MAX_STRAIN, MAX_STRAIN, size=(3, 3))
        # Apply strain to cell: new_cell = (I + strain) @ old_cell
        # Cell vectors are rows of the cell matrix
        old_cell = np.array(atoms.get_cell())
        new_cell = (np.eye(3) + strain) @ old_cell
        atoms.set_cell(new_cell, scale_atoms=False)

        # --- Apply random atomic displacements ---
        # Displacements are in Cartesian coordinates (Angstrom).
        # Each atom gets a Gaussian displacement with std = RMS for its species.
        symbols = atoms.get_chemical_symbols()
        positions = np.array(atoms.get_positions())
        for i, sym in enumerate(symbols):
            rms = RMS_MAP[sym]
            # Displacement in Cartesian Angstrom
            disp = np.random.normal(0.0, rms, size=3)
            positions[i] += disp
        atoms.set_positions(positions)

    # --- INVARIANT 1: nat and ntyp must match the actual Atoms object ---
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 4, f"Structure {idx:03d}: expected 4 atoms, got {nat}"
    assert ntyp == 2, f"Structure {idx:03d}: expected 2 species, got {ntyp}"

    out_path = f'structures/structure_{idx:03d}.cif'
    ase.io.write(out_path, atoms)

print(f"\nGenerated 51 structures: structure_000.cif .. structure_050.cif")
print(f"Temperature: {T} K")
print(f"RMS displacements: B = {RMS_B:.4f} A, N = {RMS_N:.4f} A")
print(f"Max lattice strain: {MAX_STRAIN*100:.1f}%")
