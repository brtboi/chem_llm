#!/usr/bin/env python3
"""
Generate 51 perturbed LiF structures (structure_000.cif through structure_050.cif).

structure_000.cif: unperturbed primitive cell (from LiF_primitive.cif).
structure_001..050.cif: random atomic displacements + small random lattice strain.

Temperature choice:
  LiF is stable at room temperature (melting point ~1165 C). We use T = 300 K,
  which is well within the stable phase region.

Displacement amplitudes (per-axis RMS, in Angstrom):
  For a harmonic oscillator, <u^2> = k_B*T / (M*omega^2), so lighter atoms move more.
  Li (M=6.94 amu) is lighter than F (M=19.00 amu), so Li gets a larger amplitude.
  
  Using a typical Debye temperature for LiF (~1000 K) as a rough guide:
    u_rms = sqrt(3*k_B*T / (M*omega_D^2))
  
  For Li: u_rms ~ 0.05 A (lighter, more mobile)
  For F: u_rms ~ 0.03 A (heavier, less mobile)
  
  These are per-axis RMS displacements (sqrt(U) in A^2 units).
  The crystallographic U parameter (mean-square displacement) would be:
    U_Li = (0.05)^2 = 0.0025 A^2
    U_F  = (0.03)^2 = 0.0009 A^2

Lattice strain:
  Small random strain (up to 1%) to simulate thermal expansion fluctuations.
"""

import numpy as np
import ase.io
from ase import Atoms

# Read the primitive cell
base = ase.io.read('structures/LiF_primitive.cif')
print(f"Base structure: {len(base)} atoms, symbols: {base.get_chemical_symbols()}")
print(f"Base cell:\n{base.cell}")

# Per-species RMS displacement amplitudes (in Angstrom)
# Li is lighter -> larger amplitude; F is heavier -> smaller amplitude
RMS_Li = 0.05  # A, per-axis RMS displacement for Li
RMS_F = 0.03   # A, per-axis RMS displacement for F

# Maximum random strain (fractional, i.e. 1% = 0.01)
MAX_STRAIN = 0.01

np.random.seed(42)  # reproducibility

for i in range(51):
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()
        
        # Apply random atomic displacements (Gaussian, per-axis RMS)
        for j, sym in enumerate(atoms.get_chemical_symbols()):
            if sym == 'Li':
                rms = RMS_Li
            elif sym == 'F':
                rms = RMS_F
            else:
                raise ValueError(f"Unknown species: {sym}")
            # Displace in Cartesian coordinates (Angstrom)
            atoms.positions[j] += np.random.normal(0.0, rms, 3)
        
        # Apply small random lattice strain
        # Strain tensor: symmetric, traceless-ish (small diagonal + off-diagonal)
        strain = np.random.uniform(-MAX_STRAIN, MAX_STRAIN, (3, 3))
        # Make it symmetric
        strain = 0.5 * (strain + strain.T)
        # Apply to cell: new_cell = (I + strain) @ old_cell
        new_cell = (np.eye(3) + strain) @ base.cell.array
        atoms.set_cell(new_cell, scale_atoms=True)
    
    # INVARIANT 1: nat and ntyp must match the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Expected 2 atoms, got {nat}"
    assert ntyp == 2, f"Expected 2 species, got {ntyp}"
    
    # Write the structure
    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)
    
    if i == 0:
        print(f"Wrote {out_path} (unperturbed)")
    else:
        print(f"Wrote {out_path} (perturbed)")

print("\nDone. Generated 51 structures.")
