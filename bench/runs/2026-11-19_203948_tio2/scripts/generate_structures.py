#!/usr/bin/env python3
"""
STEP 2: Generate 51 perturbed TiO2 structures (structure_000 = unperturbed base).

Perturbation model: harmonic thermal motion (Debye-Waller style) + small random
lattice strain, representative of the finite-temperature phonon ensemble.

Temperature choice: T = 300 K.
  Rutile TiO2 (mp-2657, P4_2/mnm) is the thermodynamically stable phase of TiO2
  at ambient conditions (room temperature and pressure); the anatasie/brookite
  phases are metastable at 300 K. Rutile only becomes the *less* stable phase
  above ~1800 C at high pressure, so 300 K is well within its stability range.
  At 300 K the mean-square displacements are small (U ~ 0.001-0.01 A^2), so
  the perturbed cells remain close to the ideal rutile structure.

Displacement amplitudes (per species, per Cartesian axis):
  The per-axis RMS displacement is sigma = sqrt(U), where U is the
  crystallographic mean-square displacement parameter in A^2.
  We draw displacements from a Gaussian with standard deviation sigma.

  Species amplitudes (sigma in A, i.e. RMS per axis):
    O : sigma = 0.05 A  (U = 0.0025 A^2)
        Oxygen is lighter (16 amu) and more weakly bound (terminal O in the
        rutile lattice), so it moves more.
    Ti: sigma = 0.03 A  (U = 0.0009 A^2)
        Titanium is heavier (48 amu) and sits in a stiffer octahedral site.

  These values are consistent with the experimental Debye-Waller factors for
  rutile TiO2 at 300 K (U_O ~ 0.002-0.004 A^2, U_Ti ~ 0.001-0.002 A^2).

Lattice strain: a small random symmetric strain tensor with components drawn
  from a uniform distribution in [-0.005, +0.005] (0.5% strain), applied to the
  cell vectors. This is well below the elastic limit of rutile (~1-2% for
  single-crystal rutile) and representative of thermal expansion fluctuations.

Units summary:
  - U (crystallographic displacement parameter): A^2 (mean-square)
  - sigma = sqrt(U): A (RMS per Cartesian axis)
  - Strain tensor: dimensionless (fractional change in cell vectors)
  - Cell vectors: A (ASE convention)
"""

import numpy as np
from ase.io import read, write
from ase import Atoms

# --- Configuration ---
T_K = 300  # Temperature in Kelvin (rutile is stable at ambient conditions)

# Per-species RMS displacement sigma in A (per Cartesian axis)
# sigma = sqrt(U), U in A^2
SIGMA = {
    'O': 0.05,   # A, lighter and more weakly bound
    'Ti': 0.03,  # A, heavier and in a stiffer site
}

# Strain: uniform distribution range (dimensionless)
STRAIN_RANGE = 0.005  # +/- 0.5%

N_STRUCTURES = 51  # structure_000 (unperturbed) + structure_001..050 (perturbed)

# Random seed for reproducibility
rng = np.random.default_rng(seed=42)

# --- Read base structure ---
base = read('structures/structure_base.cif')
print(f"Base structure: {base.get_chemical_formula()}, {len(base)} atoms")
print(f"Cell: a={base.cell[0,0]:.4f} A, b={base.cell[1,1]:.4f} A, c={base.cell[2,2]:.4f} A")

# Verify base structure has the expected species
base_symbols = set(base.get_chemical_symbols())
assert base_symbols == {'Ti', 'O'}, f"Unexpected species in base: {base_symbols}"

# --- Generate structures ---
for i in range(N_STRUCTURES):
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()

        # Apply random lattice strain to cell vectors
        # Strain tensor: symmetric, small, dimensionless
        # e_ij = delta_ij * (1 + strain_ij) applied to cell vectors
        strain = rng.uniform(-STRAIN_RANGE, STRAIN_RANGE, size=(3, 3))
        # Make symmetric
        strain = 0.5 * (strain + strain.T)
        # Apply to cell: new_cell = (I + strain) @ old_cell
        # In ASE, cell vectors are rows of the cell matrix
        old_cell = np.array(atoms.cell)
        new_cell = (np.eye(3) + strain) @ old_cell
        atoms.set_cell(new_cell, scale_atoms=False)

        # Apply random atomic displacements (thermal motion)
        # Displacements in Cartesian coordinates (A), Gaussian with sigma per species
        n_atoms = len(atoms)
        positions = np.array(atoms.get_positions())
        symbols = atoms.get_chemical_symbols()
        for idx, sym in enumerate(symbols):
            sigma = SIGMA[sym]
            # Draw 3D Gaussian displacement (per axis)
            disp = rng.normal(0.0, sigma, size=3)
            positions[idx] += disp
        atoms.set_positions(positions)

    # --- INVARIANT 1: nat and ntyp must be computed from the actual Atoms object ---
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 6, f"Structure {i:03d}: expected 6 atoms, got {nat}"
    assert ntyp == 2, f"Structure {i:03d}: expected 2 species, got {ntyp}"

    # Write structure
    out_path = f'structures/structure_{i:03d}.cif'
    write(out_path, atoms)
    print(f"Wrote {out_path}: {atoms.get_chemical_formula()}, {nat} atoms, {ntyp} species")

print(f"\nDone: generated {N_STRUCTURES} structures (structure_000 to structure_{N_STRUCTURES-1:03d})")
