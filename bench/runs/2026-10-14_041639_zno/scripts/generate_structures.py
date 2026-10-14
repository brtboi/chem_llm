#!/usr/bin/env python3
"""
Generate 51 perturbed ZnO structures for band-structure calculations.

Structure 000: unperturbed base (primitive cell from structures/prim.cif)
Structures 001-050: base + random thermal displacements + small random lattice strain

Temperature choice: 300 K (room temperature)
  Wurtzite ZnO is the thermodynamically stable phase at room temperature.
  The hexagonal wurtzite structure is the ground-state phase; zincblende is
  metastable. No high-temperature phase transition is needed for this phase.

Displacement model: Einstein model (independent harmonic oscillators per atom)
  RMS displacement per axis = sqrt(k_B * T / (m * omega^2))
  We use a representative Debye-like frequency for each species:
    - O (lighter, mass 16 amu): higher frequency -> smaller amplitude
    - Zn (heavier, mass 65.4 amu): lower frequency -> larger amplitude

  For ZnO at 300 K, typical Debye frequencies:
    omega_O ~ 700 cm^-1  ->  RMS_O ~ 0.045 A  (per axis)
    omega_Zn ~ 350 cm^-1 ->  RMS_Zn ~ 0.090 A (per axis)

  These are mean-square displacements U in A^2:
    U_O  = (0.045 A)^2 = 0.002025 A^2
    U_Zn = (0.090 A)^2 = 0.008100 A^2

  Per-axis RMS = sqrt(U) in A (as stated above).

Lattice strain: small random symmetric strain tensor, magnitude ~0.5%
  This represents thermal expansion / local lattice distortion.
"""

import numpy as np
import ase.io
from ase import Atoms
from ase.spacegroup import crystal
import os

# --- Constants ---
kB = 1.380649e-23      # J/K
hbar = 1.0545718e-34   # J*s
amu = 1.66053906660e-27  # kg

# Debye frequencies (rad/s) for ZnO at 300 K
# omega = 2*pi*c*nu  where nu is in cm^-1, c = 2.998e10 cm/s
# For O: nu ~ 700 cm^-1 -> omega = 2*pi*2.998e10*700 = 1.32e14 rad/s
# For Zn: nu ~ 350 cm^-1 -> omega = 2*pi*2.998e10*350 = 6.6e13 rad/s
omega_O = 2 * np.pi * 2.998e10 * 700   # rad/s
omega_Zn = 2 * np.pi * 2.998e10 * 350  # rad/s

T = 300.0  # K

# RMS displacement per axis (A)
# RMS = sqrt(k_B * T / (m * omega^2))
m_O = 16.0 * amu   # kg
m_Zn = 65.4 * amu  # kg

rms_O = np.sqrt(kB * T / (m_O * omega_O**2)) * 1e10  # convert m to A
rms_Zn = np.sqrt(kB * T / (m_Zn * omega_Zn**2)) * 1e10  # convert m to A

print(f"Temperature: {T} K")
print(f"RMS displacement O:  {rms_O:.4f} A  (U = {rms_O**2:.6f} A^2)")
print(f"RMS displacement Zn: {rms_Zn:.4f} A  (U = {rms_Zn**2:.6f} A^2)")

# --- Read base structure ---
base = ase.io.read('structures/prim.cif')
print(f"Base structure: {len(base)} atoms, species: {sorted(set(base.get_chemical_symbols()))}")

# INVARIANT 1: nat and ntyp computed from actual Atoms object
nat_base = len(base)
types_base = sorted(set(base.get_chemical_symbols()))
ntyp_base = len(types_base)
print(f"Base nat={nat_base}, ntyp={ntyp_base}")

# --- Generate structures ---
os.makedirs('structures', exist_ok=True)

rng = np.random.default_rng(seed=42)

for i in range(51):
    nnn = f"{i:03d}"
    if i == 0:
        # Unperturbed base
        atoms = base.copy()
    else:
        atoms = base.copy()
        
        # --- Random thermal displacements (Einstein model) ---
        # Displace each atom by a Gaussian random vector with per-species RMS
        pos = atoms.get_positions().copy()
        symbols = atoms.get_chemical_symbols()
        for j, sym in enumerate(symbols):
            if sym == 'O':
                rms = rms_O
            elif sym == 'Zn':
                rms = rms_Zn
            else:
                rms = 0.05  # default
            # Displacement in Cartesian coordinates (A)
            disp = rng.normal(0.0, rms, size=3)
            pos[j] += disp
        atoms.set_positions(pos)
        
        # --- Small random lattice strain (~0.5%) ---
        # Apply a small symmetric strain tensor to the cell
        # strain = [[e11, e12, e13], [e12, e22, e23], [e13, e23, e33]]
        # Magnitude ~0.5% = 0.005
        strain_mag = 0.005
        e11 = rng.uniform(-strain_mag, strain_mag)
        e22 = rng.uniform(-strain_mag, strain_mag)
        e33 = rng.uniform(-strain_mag, strain_mag)
        e12 = rng.uniform(-strain_mag, strain_mag)
        e13 = rng.uniform(-strain_mag, strain_mag)
        e23 = rng.uniform(-strain_mag, strain_mag)
        strain = np.array([
            [e11, e12, e13],
            [e12, e22, e23],
            [e13, e23, e33]
        ])
        # New cell = (I + strain) @ old_cell
        old_cell = atoms.get_cell()
        new_cell = (np.eye(3) + strain) @ old_cell
        atoms.set_cell(new_cell, scale_atoms=True)
    
    # INVARIANT 1: nat and ntyp must match the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == nat_base, f"Structure {nnn}: nat={nat} != base nat={nat_base}"
    assert ntyp == ntyp_base, f"Structure {nnn}: ntyp={ntyp} != base ntyp={ntyp_base}"
    
    out_path = f'structures/structure_{nnn}.cif'
    ase.io.write(out_path, atoms)
    
    if i < 3 or i == 50:
        print(f"  Wrote {out_path}: nat={nat}, ntyp={ntyp}")

print(f"\nGenerated 51 structures: structures/structure_000.cif through structure_050.cif")
print("Done.")
