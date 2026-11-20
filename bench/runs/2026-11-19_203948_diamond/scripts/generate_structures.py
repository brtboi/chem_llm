#!/usr/bin/env python3
"""STEP 2: Generate 51 diamond-C structures (000 unperturbed + 50 perturbed).

Perturbation model: harmonic thermal motion + small random lattice strain.

Temperature choice: 300 K.
  Diamond (Fd-3m, mp-66) is the stable phase of carbon at ambient pressure
  from cryogenic temperatures up to ~4000 K (it is kinetically stable well
  below the graphite transition, and thermodynamically the low-pressure
  phase at room temperature in practice). 300 K is therefore a temperature
  at which THIS phase is actually stable, and it is the natural reference
  temperature for comparing against Materials Project's 0 K band structure
  (thermal displacements at 300 K are small enough that the band structure
  is essentially unchanged).

Displacement amplitudes (per species, here only C):
  The crystallographic displacement parameter U is the mean-square
  displacement in A^2.  The per-axis RMS displacement is sqrt(U) in A.
  For diamond C at 300 K the experimental (Debye-Waller) mean-square
  displacement is U ~ 0.005 A^2, i.e. a per-axis RMS of sqrt(0.005) ~ 0.07 A.
  We use that value directly: each atom gets an independent Gaussian
  displacement per Cartesian axis with sigma = 0.07 A (this is the RMS,
  NOT the mean-square U).  C is the only species, so it gets its own
  amplitude; lighter or more weakly bound atoms would get a larger one.

Lattice strain: small random homogeneous strain, |eps_ij| <= 0.005
  (0.5 %), applied to the cell vectors.  This is representative of the
  lattice breathing that accompanies thermal motion without distorting
  the band structure measurably.

Units: all displacements in angstrom (ASE convention), strain dimensionless.
"""
import numpy as np
from ase.io import read, write
from ase import Atoms

BASE = 'structures/structure_000.cif'  # will be written below
N_STRUCTURES = 51

# Per-axis RMS displacement for C at 300 K (A).  U = 0.005 A^2 -> sqrt(U) = 0.07 A.
# This is the RMS (sigma of the Gaussian), not the mean-square U.
RMS_C = 0.07  # A

# Max absolute strain component (dimensionless).  0.5 % is small enough
# to keep the structure recognisably diamond while representing thermal
# lattice expansion/contraction.
MAX_STRAIN = 0.005

rng = np.random.default_rng(42)

base = read('structures/structure_base.cif')
print(f'Base structure: {base.get_chemical_formula()}, {len(base)} atoms')
print(f'Cell: {base.cell[:]}')

# Write structure_000 as the unperturbed base
write(BASE, base)

for n in range(1, N_STRUCTURES):
    atoms = base.copy()

    # Random atomic displacements: independent Gaussian per Cartesian axis,
    # sigma = RMS_C (A).  Displacements are in Cartesian coordinates.
    disp = rng.normal(0.0, RMS_C, size=(len(atoms), 3))
    atoms.positions += disp

    # Small random homogeneous lattice strain: eps is a 3x3 matrix with
    # |eps_ij| <= MAX_STRAIN.  Applied as new_cell = old_cell @ (I + eps)^T
    # (strain in the crystallographic basis).
    eps = rng.uniform(-MAX_STRAIN, MAX_STRAIN, size=(3, 3))
    # Make strain symmetric (no antisymmetric part = no rotation)
    eps = 0.5 * (eps + eps.T)
    new_cell = atoms.cell @ (np.eye(3) + eps.T)
    atoms.cell = new_cell

    # INVARIANT 1: nat and ntyp must be computed from the actual Atoms object.
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f'structure_{n:03d}: expected 2 atoms, got {nat}'
    assert ntyp == 1, f'structure_{n:03d}: expected 1 species, got {ntyp}'

    out_path = f'structures/structure_{n:03d}.cif'
    write(out_path, atoms)

print(f'Wrote {N_STRUCTURES} structures: structure_000.cif .. structure_{N_STRUCTURES-1:03d}.cif')
print('Done.')
