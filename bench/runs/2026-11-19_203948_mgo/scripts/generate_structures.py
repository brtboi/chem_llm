#!/usr/bin/env python
"""Generate 51 perturbed MgO structures (structure_000 = unperturbed base).

Perturbation model: harmonic thermal motion (Debye-Waller style) + small random
lattice strain, representative of finite-temperature phonon displacements.

Temperature choice: T = 300 K.
  MgO (periclase, rock-salt Fm-3m) is stable from room temperature up to its
  melting point (~3000 K); it is the stable phase at ambient conditions, so a
  room-temperature perturbation is physically meaningful for this phase.

Displacement amplitudes (per species, per Cartesian axis):
  The per-axis RMS displacement is sigma = sqrt(U), where U is the
  crystallographic mean-square displacement parameter in A^2.
  We estimate U from the Einstein model: U = 3 k_B T / k_spring, where
  k_spring is an effective spring constant for the atom in the lattice.
  For a rock-salt ionic crystal, a reasonable effective spring constant is
  on the order of 10-20 N/m per atom (from lattice-dynamical calculations
  of MgO: the transverse optical mode at Gamma gives k ~ 15 N/m for O,
  and Mg is lighter so its spring constant is somewhat smaller).

  Using k_B = 1.380649e-23 J/K, T = 300 K:
    k_B T = 4.142e-21 J

  For O (mass 16 amu = 2.656e-26 kg), effective spring constant ~ 15 N/m:
    U_O = 3 * 4.142e-21 / 15 = 8.28e-22 m^2 = 8.28e-6 A^2
    sigma_O = sqrt(8.28e-6) = 0.00288 A  (RMS per axis)

  For Mg (mass 24.3 amu = 4.035e-26 kg), effective spring constant ~ 12 N/m:
    (Mg is lighter and more weakly bound in the sense of lower frequency modes)
    U_Mg = 3 * 4.142e-21 / 12 = 1.04e-21 m^2 = 1.04e-5 A^2
    sigma_Mg = sqrt(1.04e-5) = 0.00323 A  (RMS per axis)

  These are small but realistic: typical Debye-Waller U values for MgO at 300 K
  are on the order of 1e-5 to 1e-4 A^2 (from neutron diffraction data).
  We use the Einstein-model estimates above as representative values.

Lattice strain: random symmetric strain tensor with components drawn from
  a uniform distribution in [-0.005, 0.005] (0.5% max strain), representing
  small elastic distortions.

Units summary:
  - U (mean-square displacement): A^2
  - sigma = sqrt(U) (RMS per-axis displacement): A
  - Strain: dimensionless
"""

import numpy as np
from ase.io import read, write
from ase import Atoms
import os

# --- Parameters ---
T = 300.0  # K
k_B = 1.380649e-23  # J/K

# Effective spring constants (N/m) per species
# O: stiffer (heavier, stronger ionic bond in TO mode) -> smaller U
# Mg: lighter, slightly weaker effective spring -> larger U
k_spring = {'Mg': 12.0, 'O': 15.0}  # N/m

# Compute U (A^2) and sigma (A) per species
# U = 3 k_B T / k_spring  [m^2] -> convert to A^2 (1 m^2 = 1e20 A^2)
sigma = {}  # RMS per-axis displacement in A
U = {}      # mean-square displacement in A^2
for sp, k in k_spring.items():
    U_m2 = 3.0 * k_B * T / k  # m^2
    U[sp] = U_m2 * 1e20       # A^2
    sigma[sp] = np.sqrt(U[sp])  # A

print(f"Temperature: {T} K")
for sp in ['Mg', 'O']:
    print(f"  {sp}: U = {U[sp]:.4e} A^2, sigma = sqrt(U) = {sigma[sp]:.5f} A (RMS per axis)")

# Strain bounds (dimensionless)
strain_max = 0.005  # 0.5%

# --- Read base structure ---
base = read('structures/structure_base.cif')
print(f"\nBase structure: {base.get_chemical_formula()}, {len(base)} atoms")
print(f"  Cell: {base.cell[:].tolist()}")

# INVARIANT 1: nat and ntyp computed from the actual Atoms object
nat_base = len(base)
typ_base = len(set(base.get_chemical_symbols()))
print(f"  nat = {nat_base}, ntyp = {typ_base}")
assert nat_base == 2, f"Expected 2 atoms in base MgO primitive cell, got {nat_base}"
assert typ_base == 2, f"Expected 2 species in base MgO, got {typ_base}"

# --- Generate structures ---
np.random.seed(42)  # reproducibility

for i in range(51):
    idx = f"{i:03d}"
    if i == 0:
        # Unperturbed base
        atoms = base.copy()
    else:
        atoms = base.copy()
        
        # 1. Random lattice strain: symmetric strain tensor
        #    e_ij uniform in [-strain_max, strain_max], symmetric
        e = np.random.uniform(-strain_max, strain_max, (3, 3))
        e = 0.5 * (e + e.T)  # symmetrize
        # New cell = (I + e) @ old_cell  (strain applied in Cartesian)
        old_cell = np.array(atoms.cell[:])
        new_cell = (np.eye(3) + e) @ old_cell
        atoms.set_cell(new_cell, scale_atoms=True)
        
        # 2. Random atomic displacements (thermal motion)
        #    Each atom gets a Gaussian displacement with per-species sigma
        #    Displacements are in Cartesian coordinates (A)
        disp = np.zeros((len(atoms), 3))
        for j, sp in enumerate(atoms.get_chemical_symbols()):
            disp[j] = np.random.normal(0.0, sigma[sp], 3)
        atoms.positions += disp
        
        # Wrap positions back into the cell (fractional coords in [0,1))
        # This keeps the structure in the standard cell representation
        frac = atoms.get_scaled_positions()
        frac = frac - np.floor(frac)  # wrap to [0, 1)
        atoms.set_scaled_positions(frac)
    
    # INVARIANT 1: assert nat/ntyp for each structure
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == nat_base, f"Structure {idx}: nat changed from {nat_base} to {nat}"
    assert ntyp == typ_base, f"Structure {idx}: ntyp changed from {typ_base} to {ntyp}"
    
    out_path = f"structures/structure_{idx}.cif"
    write(out_path, atoms)
    
    if i == 0:
        print(f"\nWrote {out_path} (unperturbed base)")
    elif i <= 3:
        # Show a few examples
        print(f"Wrote {out_path}")
        print(f"  Cell: {atoms.cell[:].tolist()}")
        print(f"  Positions (Cartesian, A): {atoms.positions.tolist()}")

print(f"\nDone. Wrote 51 structures: structure_000.cif through structure_050.cif")
print("All nat/ntyp invariants verified.")
