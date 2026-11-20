#!/usr/bin/env python3
"""STEP 2: Generate 51 Si structures (000 = unperturbed base, 001-050 = perturbed).

Perturbation model: harmonic thermal motion + small random lattice strain.

Temperature choice: T = 300 K (room temperature).
  Diamond Si is the stable phase from 0 K up to the melting point (~1687 K),
  so 300 K is well within its stability range. (Phases that exist only well
  above room temperature, e.g. high-pressure stishovite-type phases, would
  need a higher T; Si diamond does not.)

Displacement amplitudes (per species, here only Si):
  The crystallographic displacement parameter U is a MEAN-SQUARE displacement
  in A^2. The per-axis RMS displacement is sqrt(U) in A.
  We model each atom's displacement as a Gaussian with per-axis standard
  deviation sigma = sqrt(U), i.e. <dx^2> = U.

  Estimate for Si at 300 K: the Debye temperature of diamond Si is ~645 K,
  so 300 K is in the quasi-classical regime (T ~ 0.5*Theta_D). The
  experimental mean-square displacement of Si in diamond at 300 K is
  U ~ 0.0035 A^2 (from X-ray/neutron diffraction), giving a per-axis RMS
  displacement sqrt(U) ~ 0.059 A. We use U_Si = 0.0035 A^2, i.e.
  sigma_Si = sqrt(0.0035) ~ 0.0592 A per axis.
  (Only one species here; with multiple species, lighter or more weakly
  bound atoms would get larger U.)

Lattice strain: small random symmetric strain tensor, |eps_ij| <= 0.005
(0.5%), applied to the cell vectors. This is representative of the
lattice-level fluctuations a thermal ensemble would produce.

INVARIANT 1: nat/ntyp are computed from the actual ASE Atoms object for
EVERY structure and asserted, never hardcoded.
"""
import numpy as np
from ase.io import read, write
from ase import Atoms

BASE = 'structures/structure_base.cif'
N_STRUCTURES = 51          # 000 .. 050
T_KELVIN = 300.0           # temperature at which diamond Si is stable (0 K .. ~1687 K)
U_SI_A2 = 0.0035           # mean-square displacement U for Si at 300 K, in A^2
SIGMA_SI_A = np.sqrt(U_SI_A2)  # per-axis RMS displacement = sqrt(U), in A (~0.0592 A)
STRAIN_MAX = 0.005         # max |eps_ij| of the random symmetric strain tensor
SEED = 42

rng = np.random.default_rng(SEED)

base = read(BASE, format='cif')
print(f'Base: {base.get_chemical_formula()}, nat={len(base)}, '      f'ntyp={len(set(base.get_chemical_symbols()))}')
assert len(base) == 2, 'expected 2 Si atoms in MP primitive cell'

for n in range(N_STRUCTURES):
    if n == 0:
        atoms = base.copy()   # unperturbed base
    else:
        atoms = base.copy()
        # --- random atomic displacements (thermal motion) ---
        # per-axis Gaussian, std = sqrt(U) in A, applied in Cartesian coords
        disp = rng.normal(loc=0.0, scale=SIGMA_SI_A, size=atoms.positions.shape)
        atoms.positions += disp
        # --- small random symmetric lattice strain ---
        eps = rng.uniform(-STRAIN_MAX, STRAIN_MAX, size=(3, 3))
        eps = 0.5 * (eps + eps.T)          # symmetrize
        # apply strain to the cell: new_cell = (I + eps) @ old_cell (row vectors)
        old_cell = np.array(atoms.cell)
        atoms.cell = (np.eye(3) + eps) @ old_cell
    # INVARIANT 1: nat/ntyp from the actual object, asserted
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f'structure {n:03d}: nat={nat}, expected 2'
    assert ntyp == 1, f'structure {n:03d}: ntyp={ntyp}, expected 1'
    out = f'structures/structure_{n:03d}.cif'
    write(out, atoms, format='cif')
print(f'Wrote {N_STRUCTURES} structures: structure_000.cif .. structure_{N_STRUCTURES-1:03d}.cif')
print(f'T = {T_KELVIN} K, U_Si = {U_SI_A2} A^2, sigma_Si = sqrt(U) = {SIGMA_SI_A:.4f} A/axis, strain <= {STRAIN_MAX}')
