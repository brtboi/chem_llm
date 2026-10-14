#!/usr/bin/env python3
"""
Generate 51 perturbed PbTe structures (structure_000.cif through structure_050.cif).

Structure 000 is the unperturbed primitive cell (already written by reduce_cell.py).
Structures 001-050 are perturbed copies with:
  - Random atomic displacements (thermal motion at T = 300 K)
  - Small random lattice strain

Temperature choice: T = 300 K (room temperature)
  PbTe is a stable rock-salt semiconductor at room temperature (band gap ~0.8 eV).
  It is not a high-temperature-only phase, so 300 K is appropriate.

Displacement amplitudes (Einstein model):
  U = 3*k_B*T / (M * omega^2)  [A^2, mean-square displacement]
  RMS per-axis displacement = sqrt(U) [A]

  For rock-salt PbTe, typical phonon frequency ~ 5 THz (omega = 2*pi*5e12 rad/s).
  k_B = 1.380649e-23 J/K, T = 300 K

  Pb: M = 207.2 g/mol = 207.2e-3 kg/mol
       U_Pb = 3 * 1.380649e-23 * 300 / (207.2e-3 * (2*pi*5e12)^2) / 6.022e23
            = 3 * 1.380649e-23 * 300 / (207.2e-3 * 9.8696e25) / 6.022e23
            = 1.2426e-20 / (2.045e22) / 6.022e23
            = 6.076e-43 / 6.022e23  -- wait, let me redo this carefully.

  Actually: U = 3*k_B*T / (M_atom * omega^2) where M_atom is mass of one atom in kg.
  M_Pb_atom = 207.2e-3 / 6.022e23 = 3.441e-25 kg
  omega = 2*pi*5e12 = 3.1416e13 rad/s
  omega^2 = 9.8696e26 (rad/s)^2
  U_Pb = 3 * 1.380649e-23 * 300 / (3.441e-25 * 9.8696e26)
       = 1.2426e-20 / (3.396e2)
       = 3.659e-23 m^2 = 3.659e-13 A^2  -- that's way too small.

  Let me use a more standard approach. For a typical ionic solid at 300 K:
  RMS displacement ~ 0.05-0.15 A for heavy atoms.
  
  Using Debye-Waller factors from literature for PbTe at 300 K:
  B_Pb ~ 0.5 A^2, B_Te ~ 0.7 A^2 (B = 8*pi^2*U, so U = B/(8*pi^2))
  U_Pb = 0.5 / (8*pi^2) = 0.00633 A^2  -> RMS = sqrt(0.00633) = 0.0796 A
  U_Te = 0.7 / (8*pi^2) = 0.00886 A^2  -> RMS = sqrt(0.00886) = 0.0941 A

  These are reasonable for a heavy ionic semiconductor at 300 K.
  Te is lighter (128 amu vs 207 amu) so it has a larger displacement.

Lattice strain: small random strain tensor with components ~ 0.1% (1e-3)
  This represents typical thermal expansion / local strain fluctuations.
"""

import numpy as np
import ase.io
from ase import Atoms

# Read the base primitive cell (structure_000.cif)
base = ase.io.read('structures/structure_000.cif')
print(f"Base structure: {len(base)} atoms, symbols: {base.get_chemical_symbols()}")

# Displacement parameters (RMS per-axis, in Angstrom)
# Derived from Debye-Waller factors for PbTe at 300 K (see docstring)
# U is mean-square displacement in A^2, RMS = sqrt(U) in A
RMS_Pb = 0.0796  # A, per-axis RMS displacement for Pb
RMS_Te = 0.0941  # A, per-axis RMS displacement for Te (lighter, moves more)

# Strain magnitude: 0.1% random strain
STRAIN_MAG = 1e-3  # dimensionless

# Number of structures to generate (001 through 050)
N_STRUCTURES = 50

# Set random seed for reproducibility
np.random.seed(42)

for i in range(1, N_STRUCTURES + 1):
    nnn = f"{i:03d}"
    atoms = base.copy()
    
    # Apply random lattice strain
    # Generate a small random symmetric strain tensor
    strain = np.random.randn(3, 3) * STRAIN_MAG
    strain = (strain + strain.T) / 2  # make symmetric
    
    # Apply strain to the cell
    # new_cell = (I + strain) @ old_cell
    old_cell = atoms.get_cell()
    new_cell = (np.eye(3) + strain) @ old_cell
    atoms.set_cell(new_cell, scale_atoms=True)  # scale atoms with cell
    
    # Apply random atomic displacements (thermal motion)
    # Displacements in Cartesian coordinates (Angstrom)
    # Each atom gets a random Gaussian displacement with species-specific RMS
    symbols = atoms.get_chemical_symbols()
    displacements = np.zeros((len(atoms), 3))
    for idx, sym in enumerate(symbols):
        if sym == 'Pb':
            rms = RMS_Pb
        elif sym == 'Te':
            rms = RMS_Te
        else:
            raise ValueError(f"Unknown species: {sym}")
        displacements[idx] = np.random.randn(3) * rms
    
    atoms.positions += displacements
    
    # INVARIANT 1: nat and ntyp must match the actual structure
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Structure {nnn}: expected 2 atoms, got {nat}"
    assert ntyp == 2, f"Structure {nnn}: expected 2 species, got {ntyp}"
    
    out_path = f'structures/structure_{nnn}.cif'
    ase.io.write(out_path, atoms)
    print(f"Wrote {out_path}: {nat} atoms, {ntyp} species")

print(f"\nDone. Generated {N_STRUCTURES} perturbed structures (001-{N_STRUCTURES:03d}).")
print("Structure 000 (unperturbed) was already written by reduce_cell.py.")
