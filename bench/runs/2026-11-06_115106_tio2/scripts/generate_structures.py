#!/usr/bin/env python
"""
Generate 51 perturbed TiO2 structures (structure_000.cif through structure_050.cif).

Structure 000 is the unperturbed primitive rutile cell (6 atoms: 2 Ti, 4 O).
Structures 001-050 are perturbed copies with:
  - random atomic displacements (thermal motion)
  - small random lattice strain

Temperature choice:
  Rutile TiO2 is the stable phase above ~860 K (1133 C) at ambient pressure.
  Below that, the metastable anatase phase is more stable. We choose T = 1000 K,
  which is well within the rutile stability range and gives physically meaningful
  thermal displacements.

Displacement amplitudes (RMS, in Angstrom):
  The mean-square displacement U (in A^2) for a harmonic oscillator at temperature T is:
    U = 3 * k_B * T / (M * omega^2)
  where omega is the characteristic vibrational frequency.
  
  For a rough estimate, we use the Debye model:
    U = (9 * k_B * T) / (4 * M * omega_D^2) * f(T/T_D)
  
  For TiO2 at 1000 K, typical RMS displacements are:
    - O (lighter, more weakly bound): ~0.10 A  (U_O = 0.010 A^2)
    - Ti (heavier, more strongly bound): ~0.05 A  (U_Ti = 0.0025 A^2)
  
  These values are consistent with experimental neutron diffraction data for
  rutile TiO2 at elevated temperatures. The per-axis RMS displacement is
  sqrt(U/3) for isotropic motion, but we use the full RMS (sqrt(U)) as the
  standard deviation for each Cartesian displacement component, which is
  equivalent to assuming the displacement is distributed over 3 axes with
  total variance U.

  Actually, let's be more careful. If U is the total mean-square displacement
  (sum over 3 axes), then the per-axis variance is U/3, and the per-axis RMS
  is sqrt(U/3). We'll use this convention:
    - O: U = 0.010 A^2, per-axis RMS = sqrt(0.010/3) = 0.0577 A
    - Ti: U = 0.0025 A^2, per-axis RMS = sqrt(0.0025/3) = 0.0289 A

Lattice strain:
  Small random strain (up to ~1%) to simulate thermal expansion fluctuations.
  This is applied as a random symmetric strain tensor with components in [-0.01, 0.01].
"""

import numpy as np
import ase.io
from ase import Atoms

# Read the base structure (structure_000.cif is the primitive cell)
base = ase.io.read('structures/structure_000.cif')
print(f"Base structure: {len(base)} atoms, symbols: {base.get_chemical_symbols()}")

# Temperature and displacement parameters
T = 1000.0  # K, rutile is stable above ~860 K

# Mean-square displacements U in A^2 (total, sum over 3 axes)
U_O = 0.010   # A^2 for oxygen
U_Ti = 0.0025 # A^2 for titanium

# Per-axis RMS displacement = sqrt(U/3) in A
rms_O = np.sqrt(U_O / 3.0)   # ~0.0577 A
rms_Ti = np.sqrt(U_Ti / 3.0) # ~0.0289 A

print(f"Temperature: {T} K")
print(f"O: U = {U_O} A^2, per-axis RMS = {rms_O:.4f} A")
print(f"Ti: U = {U_Ti} A^2, per-axis RMS = {rms_Ti:.4f} A")

# Random seed for reproducibility
np.random.seed(42)

# Structure 000 is already written (unperturbed base)
# Generate structures 001-050
for i in range(1, 51):
    atoms = base.copy()
    
    # Apply random lattice strain (symmetric strain tensor, components in [-0.01, 0.01])
    strain = np.random.uniform(-0.01, 0.01, size=(3, 3))
    # Make it symmetric
    strain = 0.5 * (strain + strain.T)
    # Apply strain to the cell
    new_cell = atoms.cell * (np.eye(3) + strain)
    atoms.set_cell(new_cell, scale_atoms=True)
    
    # Apply random atomic displacements
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    for j, sym in enumerate(symbols):
        if sym == 'O':
            disp = np.random.normal(0.0, rms_O, size=3)
        elif sym == 'Ti':
            disp = np.random.normal(0.0, rms_Ti, size=3)
        else:
            raise ValueError(f"Unknown element: {sym}")
        positions[j] += disp
    
    atoms.set_positions(positions)
    
    # INVARIANT 1: nat and ntyp must match the actual structure
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 6, f"Structure {i:03d}: expected 6 atoms, got {nat}"
    assert ntyp == 2, f"Structure {i:03d}: expected 2 types, got {ntyp}"
    
    # Write the structure
    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)
    
    if i % 10 == 0:
        print(f"  Wrote {out_path}")

print("Done. Generated structures 001-050.")
