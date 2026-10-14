#!/usr/bin/env python3
"""
Generate 51 perturbed Bi2Se3 structures (structure_000.cif through structure_050.cif).

Structure 000: unperturbed primitive cell (from structures/primitive_Bi2Se3.cif).
Structures 001-050: random atomic displacements (thermal motion) + small random lattice strain.

Temperature choice: 300 K (room temperature).
Bi2Se3 is a room-temperature topological insulator with no high-temperature phase transition.
It is stable at ambient conditions, so 300 K is the natural choice.

Displacement amplitudes (RMS, in Angstrom):
  - Se (79 amu, lighter): RMS = 0.09 A  -> U = 0.0081 A^2 (mean-square displacement)
  - Bi (209 amu, heavier): RMS = 0.06 A  -> U = 0.0036 A^2
  Justification: lighter atoms vibrate with larger amplitude at the same temperature.
  These values are consistent with typical Debye-Waller factors for Bi2Se3 at 300 K.

Lattice strain: random strain tensor with components drawn from N(0, 0.005),
  i.e. ~0.5% RMS strain per component. This is small enough to keep the structure
  in the same phase but large enough to sample different local environments.

Units: all displacements in Angstrom, strain dimensionless.
"""

import numpy as np
import ase.io
from ase import Atoms

# Read the primitive cell (already reduced in STEP 1)
base = ase.io.read('structures/primitive_Bi2Se3.cif')

# INVARIANT 1: nat and ntyp computed from the actual Atoms object
nat = len(base)
types = set(base.get_chemical_symbols())
ntyp = len(types)
print(f"Base structure: nat={nat}, ntyp={ntyp}, species={types}")
assert nat == 5, f"Expected 5 atoms in primitive Bi2Se3, got {nat}"
assert ntyp == 2, f"Expected 2 species, got {ntyp}"

# Displacement parameters (RMS in Angstrom)
# U is the mean-square displacement in A^2; RMS = sqrt(U)
RMS_SE = 0.09  # Angstrom, for Se (lighter)
RMS_BI = 0.06  # Angstrom, for Bi (heavier)

# Strain: RMS per component
STRAIN_RMS = 0.005  # dimensionless, ~0.5%

np.random.seed(42)  # reproducibility

for i in range(51):
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()
        
        # Apply random lattice strain
        # Strain tensor: symmetric 3x3, each component ~ N(0, STRAIN_RMS)
        strain = np.random.normal(0, STRAIN_RMS, (3, 3))
        # Make it symmetric (physical strain tensor is symmetric)
        strain = 0.5 * (strain + strain.T)
        # New cell = (I + strain) @ old_cell
        old_cell = atoms.get_cell()
        new_cell = (np.eye(3) + strain) @ old_cell
        atoms.set_cell(new_cell, scale_atoms=True)
        
        # Apply random atomic displacements (thermal motion)
        # Each atom gets a random displacement drawn from N(0, RMS_species^2)
        symbols = atoms.get_chemical_symbols()
        for j, sym in enumerate(symbols):
            if sym == 'Se':
                rms = RMS_SE
            elif sym == 'Bi':
                rms = RMS_BI
            else:
                raise ValueError(f"Unexpected species: {sym}")
            disp = np.random.normal(0, rms, 3)  # in Angstrom
            atoms.positions[j] += disp
    
    # INVARIANT 1: verify nat/ntyp for each structure
    assert len(atoms) == nat, f"Structure {i:03d}: nat changed from {nat} to {len(atoms)}"
    assert len(set(atoms.get_chemical_symbols())) == ntyp, \
        f"Structure {i:03d}: ntyp changed from {ntyp}"
    
    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)
    
    if i == 0:
        print(f"Wrote {out_path} (unperturbed)")
    else:
        print(f"Wrote {out_path} (perturbed)")

print(f"\nDone. Generated 51 structures in structures/.")
