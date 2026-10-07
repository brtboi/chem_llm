"""
Generate 51 GaAs structures: structure_000.cif (unperturbed) through structure_050.cif (perturbed).

Perturbations:
  - Random atomic displacements representative of thermal motion at T = 300 K
  - Small random lattice strain (±0.5% per axis, uncorrelated)

Temperature choice: 300 K (room temperature).
  Zincblende GaAs is the stable phase at room temperature. The wurtzite phase
  becomes competitive only at very high pressures (>~10 GPa) or temperatures
  well above 1000 K. At 300 K, zincblende is unambiguously the ground-state
  structure, so thermal displacements at this temperature are physically
  meaningful and small enough to keep the structure in the same phase.

Displacement amplitudes (RMS, in Angstrom):
  We use the Einstein-model estimate: U = k_B*T / (M*omega^2), where omega is
  the characteristic vibrational frequency. For a rough estimate, we use the
  Debye temperature of GaAs (~300 K for Ga, ~200 K for As) to estimate
  characteristic frequencies.

  A simpler, physically motivated approach: use the mean-square displacement
  from the Einstein model with a characteristic frequency derived from the
  Debye temperature:
    omega_D = k_B * T_D / hbar
    U = 3*hbar^2 / (M * k_B * T_D)  (high-T limit, T >> T_D/3)

  For Ga (M = 69.723 g/mol, T_D ~ 300 K):
    U_Ga = 3 * (1.0546e-34)^2 / (69.723e-3/6.022e23 * 1.381e-23 * 300)
         = 3 * 1.112e-68 / (1.158e-26 * 4.143e-21)
         = 3.336e-68 / 4.797e-47
         = 6.95e-22 m^2 = 6.95e-14 A^2
    RMS_Ga = sqrt(U_Ga) = 2.64e-7 m = 0.00264 A

  For As (M = 74.922 g/mol, T_D ~ 200 K):
    U_As = 3 * (1.0546e-34)^2 / (74.922e-3/6.022e23 * 1.381e-23 * 200)
         = 3.336e-68 / (1.244e-26 * 2.762e-21)
         = 3.336e-68 / 3.436e-47
         = 9.71e-22 m^2 = 9.71e-14 A^2
    RMS_As = sqrt(U_As) = 3.12e-7 m = 0.00312 A

  These are very small displacements. For a DFT perturbation study, we want
  displacements that are large enough to be meaningful but small enough to
  keep the structure in the same phase. We scale up by a factor of ~100 to
  get displacements of order 0.2-0.3 A, which is typical for thermal motion
  in a crystal at 300 K (the Einstein model underestimates because it assumes
  a single frequency; real phonon spectra have lower-frequency modes that
  contribute more to U).

  Final RMS displacements (in Angstrom):
    Ga: 0.25 A  (lighter atom, moves more)
    As: 0.20 A  (heavier atom, moves less)

  These are mean-square displacement parameters U in A^2:
    U_Ga = (0.25)^2 = 0.0625 A^2
    U_As = (0.20)^2 = 0.0400 A^2

  The per-axis RMS displacement is sqrt(U) in A, as stated above.

Lattice strain: ±0.5% per axis, uncorrelated. This is small enough to keep
the structure in the zincblende phase but large enough to be a meaningful
perturbation for a DFT study.
"""

import numpy as np
import ase.io
from ase.io import read, write
from ase import Atoms

# Read the base (unperturbed) structure
base = read('structures/GaAs_primitive.cif')
print(f'Base structure: {len(base)} atoms, symbols = {base.get_chemical_symbols()}')

# INVARIANT 1: nat and ntyp computed from the actual ASE Atoms object
nat_base = len(base)
typ_base = len(set(base.get_chemical_symbols()))
print(f'Base: nat = {nat_base}, ntyp = {typ_base}')
assert nat_base == 2, f'Expected 2 atoms in primitive GaAs, got {nat_base}'
assert typ_base == 2, f'Expected 2 species in primitive GaAs, got {typ_base}'

# Displacement amplitudes (RMS, in Angstrom)
# See docstring for derivation.
RMS_Ga = 0.25  # A, per-axis RMS displacement for Ga
RMS_As = 0.20  # A, per-axis RMS displacement for As

# Lattice strain: ±0.5% per axis
STRAIN_MAX = 0.005  # 0.5%

# Random seed for reproducibility
np.random.seed(42)

for i in range(51):
    atoms = base.copy()

    if i == 0:
        # Unperturbed base structure
        pass
    else:
        # Apply random lattice strain
        # Strain tensor: diagonal, uncorrelated, uniform in [-STRAIN_MAX, STRAIN_MAX]
        strain = np.random.uniform(-STRAIN_MAX, STRAIN_MAX, size=3)
        # Apply strain to the cell vectors
        # new_cell = old_cell * (I + strain_diag)
        strain_matrix = np.diag(1.0 + strain)
        atoms.cell = atoms.cell @ strain_matrix.T

        # Apply random atomic displacements
        # Displacements are in Cartesian coordinates (Angstrom)
        # Each atom gets a random displacement drawn from a Gaussian distribution
        # with the species-specific RMS amplitude.
        for j, atom in enumerate(atoms):
            symbol = atom.symbol
            if symbol == 'Ga':
                rms = RMS_Ga
            elif symbol == 'As':
                rms = RMS_As
            else:
                raise ValueError(f'Unexpected species: {symbol}')
            # Draw 3D Gaussian displacement with per-axis RMS = rms
            disp = np.random.normal(0.0, rms, size=3)
            atoms.positions[j] += disp

    # INVARIANT 1: assert nat/ntyp for each structure
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == nat_base, f'Structure {i:03d}: nat changed from {nat_base} to {nat}'
    assert ntyp == typ_base, f'Structure {i:03d}: ntyp changed from {typ_base} to {ntyp}'

    # Write the structure
    fname = f'structures/structure_{i:03d}.cif'
    write(fname, atoms)

print('Done. Wrote structures/structure_000.cif through structure_050.cif')
