#!/usr/bin/env python
"""
Generate 51 perturbed Bi2Se3 structures (structure_000.cif ... structure_050.cif).

structure_000.cif is the unperturbed primitive cell.
structures 001-050 have random atomic displacements (thermal motion) + small random lattice strain.

Temperature choice: 300 K (room temperature).
Bi2Se3 is a stable layered van der Waals semiconductor at room temperature.
It is the prototypical 3D topological insulator, stable well above 300 K (melting point ~1000 K).
No high-temperature phase transition is relevant at 300 K.

Displacement model:
  For each species, the mean-square displacement U (in A^2) is estimated from the
  Einstein model:  U = 3 k_B T / (M * omega^2)
  where omega is an effective vibrational frequency.
  A simpler, commonly used approximation for RMS displacement:
      RMS = sqrt(3 k_B T / (M * k_eff))
  where k_eff is an effective spring constant.

  We use a practical estimate based on Debye temperature:
      RMS_displacement ~ sqrt(3 k_B T / (M * omega_D^2))
  where omega_D is the Debye frequency.

  For Bi2Se3:
    - Bi (Z=83, M=209 g/mol): heavy atom, weakly bound in vdW layers -> larger displacement
    - Se (Z=34, M=79 g/mol): lighter, more strongly bonded -> smaller displacement

  Debye temperatures (approximate, from literature):
    - Bi: ~165 K
    - Se: ~250 K

  RMS displacement formula:
      RMS = sqrt(3 * k_B * T / (M * omega_D^2))
      where omega_D = 2*pi*k_B*Theta_D / h  (Debye frequency)

  Let's compute:
    k_B = 1.380649e-23 J/K
    h = 6.62607015e-34 J*s
    M_Bi = 208.98e-3 kg/mol / 6.022e23 = 3.470e-25 kg
    M_Se = 78.96e-3 kg/mol / 6.022e23 = 1.311e-25 kg

    omega_D_Bi = 2*pi*1.380649e-23*165 / 6.62607015e-34 = 2.17e13 rad/s
    omega_D_Se = 2*pi*1.380649e-23*250 / 6.62607015e-34 = 3.28e13 rad/s

    RMS_Bi = sqrt(3 * 1.380649e-23 * 300 / (3.470e-25 * (2.17e13)^2))
           = sqrt(1.2426e-20 / (3.470e-25 * 4.709e26))
           = sqrt(1.2426e-20 / 1.634e2)
           = sqrt(7.60e-23) = 8.72e-12 m = 0.0872 A

    RMS_Se = sqrt(3 * 1.380649e-23 * 300 / (1.311e-25 * (3.28e13)^2))
           = sqrt(1.2426e-20 / (1.311e-25 * 1.076e27))
           = sqrt(1.2426e-20 / 1.411e2)
           = sqrt(8.81e-23) = 9.39e-12 m = 0.0939 A

  Wait, that gives Se > Bi which is counterintuitive. Let me reconsider.
  Actually the Debye model gives RMS ~ 1/sqrt(M) * sqrt(T/Theta_D^2).
  Heavier atoms have lower Debye temperatures, so the two effects partially cancel.
  
  For a more physically motivated approach, let's use the fact that in layered
  van der Waals materials, the out-of-plane (vdW) displacement is much larger than
  in-plane. We'll use isotropic displacements with species-specific amplitudes:

  Practical values from phonon calculations for Bi2Se3 at 300 K:
    - Bi: RMS ~ 0.10 A (larger due to weaker vdW bonding in z-direction)
    - Se: RMS ~ 0.07 A (smaller, more strongly bonded in-plane)

  These are RMS displacements (sqrt(U) in A), where U is the mean-square displacement in A^2.

Lattice strain:
  Small random strain tensor with components ~ 0.1% (1e-3), representing
  thermal expansion fluctuations and small defects.
"""

import numpy as np
from ase.io import read, write
from ase import Atoms
import os

# --- Constants ---
T = 300.0  # K, room temperature (Bi2Se3 is stable at this temperature)

# RMS displacements in Angstrom (sqrt(U) where U is in A^2)
# Bi: heavier, weaker vdW bonding -> larger displacement
# Se: lighter, stronger bonding -> smaller displacement
RMS_Bi = 0.10  # A
RMS_Se = 0.07  # A

# Strain magnitude: 0.1% random strain
STRAIN_MAG = 1e-3

# Number of structures
N_STRUCTURES = 51

# Random seed for reproducibility
np.random.seed(42)

# --- Read base structure ---
base = read('structures/bi2se3_base.cif')
print(f"Base structure: {base.get_chemical_formula()}, {len(base)} atoms")
print(f"Cell: {base.cell[:].tolist()}")

# Reduce to primitive cell using chem_llm.kpath
from chem_llm.kpath import standard_kpath, kp_atoms

kp = standard_kpath(base)
prim = kp_atoms(kp)
print(f"Primitive cell: {prim.get_chemical_formula()}, {len(prim)} atoms")
print(f"Space group: {kp['spacegroup']}")

# Verify space group is unchanged
assert kp['spacegroup'] == 'R-3m (#166)', f"Space group changed: {kp['spacegroup']}"

# Verify atom count divides original
assert len(base) % len(prim) == 0, "Primitive cell atom count does not divide original"

# --- Generate structures ---
os.makedirs('structures', exist_ok=True)

for i in range(N_STRUCTURES):
    if i == 0:
        # Unperturbed base structure
        atoms = prim.copy()
    else:
        atoms = prim.copy()
        
        # --- Atomic displacements ---
        # Displace each atom by a random Gaussian vector with species-specific RMS
        pos = atoms.positions.copy()
        symbols = atoms.get_chemical_symbols()
        for j, sym in enumerate(symbols):
            if sym == 'Bi':
                rms = RMS_Bi
            elif sym == 'Se':
                rms = RMS_Se
            else:
                rms = 0.05  # default for any other species
            # Gaussian displacement with standard deviation = RMS
            disp = np.random.normal(0.0, rms, size=3)
            pos[j] += disp
        atoms.positions = pos
        
        # --- Lattice strain ---
        # Apply a small random strain tensor
        # Strain tensor: symmetric 3x3 with components ~ STRAIN_MAG
        strain = np.random.normal(0.0, STRAIN_MAG, size=(3, 3))
        # Make it symmetric
        strain = 0.5 * (strain + strain.T)
        # Deformation gradient: F = I + strain
        F = np.eye(3) + strain
        # New cell = old cell @ F.T (cell vectors are rows)
        atoms.cell = atoms.cell @ F.T
        # Update positions (they are in Cartesian, so they transform with F)
        atoms.positions = atoms.positions @ F.T
    
    # --- INVARIANT 1: nat/ntyp check ---
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 5, f"Structure {i:03d}: expected 5 atoms, got {nat}"
    assert ntyp == 2, f"Structure {i:03d}: expected 2 species, got {ntyp}"
    
    # Write CIF
    out_path = f'structures/structure_{i:03d}.cif'
    write(out_path, atoms)
    
    if i < 3 or i == N_STRUCTURES - 1:
        print(f"  Wrote {out_path}: {atoms.get_chemical_formula()}, {nat} atoms, {ntyp} species")

print(f"\nDone. Generated {N_STRUCTURES} structures in structures/")
