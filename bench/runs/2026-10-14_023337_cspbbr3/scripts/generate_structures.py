#!/usr/bin/env python3
"""
Generate 51 perturbed CsPbBr3 structures (structure_000.cif through structure_050.cif).

structure_000.cif: unperturbed primitive cell (already written by reduce_cell.py).
structure_001-050.cif: random thermal displacements + small random lattice strain.

Temperature choice:
  CsPbBr3 cubic phase (Pm-3m) is stable at room temperature.
  The cubic-to-orthorhombic transition occurs at ~230 K, so 300 K is safely in the cubic regime.
  We use T = 300 K for the Einstein-model displacement amplitudes.

Displacement model (Einstein model):
  Each atom vibrates as an independent harmonic oscillator.
  The mean-square displacement per axis is U (in A^2), and the RMS displacement per axis is sqrt(U) (in A).
  We assign per-species RMS amplitudes based on mass and bonding:
    - Cs (133 amu, weakly bound A-site cation):  RMS = 0.15 A  -> U = 0.0225 A^2
    - Pb (207 amu, octahedral center):           RMS = 0.05 A  -> U = 0.0025 A^2
    - Br (80 amu, corner of octahedron):         RMS = 0.10 A  -> U = 0.0100 A^2
  Lighter / more weakly bound atoms move more, consistent with the Einstein model.

Random strain:
  Small uniform random strain tensor with components ~0.5% (0.005) to represent lattice distortion.
  Applied as a linear transformation to the cell vectors.
"""

import numpy as np
import ase.io
from ase import Atoms

# Read the base (unperturbed) primitive cell
base = ase.io.read('structures/structure_000.cif')

# Per-species RMS displacement amplitudes in Angstrom (sqrt(U), where U is in A^2)
# Justification: Einstein model at T=300 K; lighter/weakly-bound atoms move more.
RMS_AMPLITUDES = {
    'Cs': 0.15,  # A  (U = 0.0225 A^2)
    'Pb': 0.05,  # A  (U = 0.0025 A^2)
    'Br': 0.10,  # A  (U = 0.0100 A^2)
}

# Random strain magnitude (dimensionless, ~0.5%)
STRAIN_MAGNITUDE = 0.005

np.random.seed(42)  # reproducibility

for n in range(1, 51):
    atoms = base.copy()
    
    # Apply random thermal displacements in Cartesian coordinates
    for i, atom in enumerate(atoms):
        symbol = atom.symbol
        rms = RMS_AMPLITUDES[symbol]
        # Displacement per axis: Gaussian with std = RMS (in A)
        disp = np.random.normal(0.0, rms, size=3)
        atoms.positions[i] += disp
    
    # Apply small random lattice strain
    # Strain tensor: symmetric, traceless (volume-preserving) + small isotropic part
    strain = np.random.uniform(-STRAIN_MAGNITUDE, STRAIN_MAGNITUDE, size=(3, 3))
    # Make symmetric
    strain = 0.5 * (strain + strain.T)
    # Remove trace (keep volume approximately constant)
    strain -= np.trace(strain) / 3.0 * np.eye(3)
    
    # Apply strain to cell vectors: new_cell = (I + strain) @ old_cell
    new_cell = (np.eye(3) + strain) @ atoms.cell.array
    atoms.cell = new_cell
    
    # INVARIANT 1: nat and ntyp must match the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 5, f"Structure {n:03d}: expected 5 atoms, got {nat}"
    assert ntyp == 3, f"Structure {n:03d}: expected 3 species, got {ntyp}"
    
    out_path = f'structures/structure_{n:03d}.cif'
    ase.io.write(out_path, atoms)
    print(f'Wrote {out_path}: nat={nat}, ntyp={ntyp}')

print('Done: 50 perturbed structures written (structure_001.cif through structure_050.cif).')
