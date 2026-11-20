#!/usr/bin/env python
"""
STEP 2: Generate 51 perturbed Al structures (000 = unperturbed base, 001-050 perturbed).

Temperature choice: T = 600 K
  - Al is a stable fcc solid from 0 K up to its melting point 933 K (1660 F).
  - 600 K is a realistic high-temperature regime where thermal motion is
    significant but the crystal is far from melting, so the fcc structure
    remains stable. This is well above room temperature (300 K) so the
    displacements are clearly visible, yet safely below the melting point.

Debye-Waller displacement parameters for Al at 600 K:
  - The mean-square displacement U (crystallographic Debye-Waller parameter)
    for Al at 600 K is approximately 0.0015 A^2.
  - The per-axis RMS displacement is sqrt(U) = sqrt(0.0015) ~ 0.0387 A.
  - We use Gaussian displacements with sigma = sqrt(U) per axis.
  - Units: U is in A^2 (mean-square displacement), sqrt(U) is in A (RMS per axis).

Lattice strain: small random isotropic + anisotropic strain, magnitude ~0.5%.
  - This represents thermal expansion / elastic fluctuations.

INVARIANT 1: nat and ntyp are computed from the actual ASE Atoms object,
  never hardcoded.
"""

import os
import numpy as np
from ase.io import read, write
from ase import Atoms

# --- Configuration ---
T_K = 600  # Temperature in Kelvin (Al stable up to 933 K)

# Debye-Waller mean-square displacement U for Al at 600 K, in A^2
# (from literature / Debye model estimate for fcc Al)
U_Al_A2 = 0.0015  # A^2
sigma_Al_A = np.sqrt(U_Al_A2)  # A, per-axis RMS displacement

# Random lattice strain magnitude (fractional, ~0.5%)
strain_mag = 0.005

# Number of structures to generate (000 = base, 001-050 = perturbed)
N_STRUCTURES = 51

# Base structure path
BASE_CIF = 'structures/Al_base.cif'
OUT_DIR = 'structures'

# Random seed for reproducibility
np.random.seed(42)


def apply_random_strain(atoms, strain_mag):
    """
    Apply a small random lattice strain to the structure.
    
    The strain tensor is a small random symmetric matrix with components
    of order strain_mag. This represents thermal expansion / elastic
    fluctuations of the lattice.
    
    The new cell = old_cell @ (I + strain)^T  (strain acts on fractional coords)
    In ASE, atoms.cell is a 3x3 matrix of lattice vectors (rows).
    To apply strain to the lattice vectors: new_cell = (I + strain) @ old_cell
    where strain is a small 3x3 matrix.
    """
    # Generate a small random symmetric strain tensor
    # Components are of order strain_mag
    strain = np.random.randn(3, 3) * strain_mag
    # Make it symmetric (physical strain tensor is symmetric)
    strain = 0.5 * (strain + strain.T)
    
    # Apply to cell: new_cell = (I + strain) @ old_cell
    # old_cell rows are lattice vectors a, b, c
    # (I + strain) @ old_cell transforms each lattice vector
    new_cell = (np.eye(3) + strain) @ atoms.cell
    
    # Create new Atoms with the strained cell, keeping fractional positions
    new_atoms = Atoms(
        symbols=atoms.get_chemical_symbols(),
        scaled_positions=atoms.get_scaled_positions(),
        cell=new_cell,
        pbc=True,
    )
    return new_atoms


def apply_random_displacements(atoms, sigma_per_species):
    """
    Apply random Gaussian displacements to each atom.
    
    sigma_per_species: dict mapping element symbol to sigma in A (per-axis RMS).
    Displacements are in Cartesian coordinates (A).
    """
    positions = atoms.get_positions().copy()  # in A
    symbols = atoms.get_chemical_symbols()
    
    for i, sym in enumerate(symbols):
        sigma = sigma_per_species.get(sym, 0.0)
        if sigma > 0:
            # Gaussian displacement in each Cartesian direction, sigma in A
            positions[i] += np.random.normal(0.0, sigma, size=3)
    
    new_atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=atoms.cell,
        pbc=True,
    )
    return new_atoms


def main():
    # Read base structure
    base = read(BASE_CIF)
    print(f"Base structure: {base.get_chemical_formula()} "
          f"nat={len(base)}, ntyp={len(set(base.get_chemical_symbols()))}")
    print(f"Cell: {base.cell[:].tolist()}")
    print(f"Space group info from CIF: {base.info.get('space_group', 'N/A')}")
    
    # Per-species displacement amplitudes (sigma in A, per-axis RMS)
    # Al is the only species; lighter/weakly-bound atoms move more.
    # At 600 K, sigma_Al = sqrt(U) = sqrt(0.0015 A^2) ~ 0.0387 A
    sigma_per_species = {'Al': sigma_Al_A}
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    for n in range(N_STRUCTURES):
        nnn = f"{n:03d}"
        out_path = os.path.join(OUT_DIR, f"structure_{nnn}.cif")
        
        if n == 0:
            # Unperturbed base structure
            atoms = base.copy()
        else:
            # Perturbed: random displacements + small random lattice strain
            atoms = base.copy()
            atoms = apply_random_strain(atoms, strain_mag)
            atoms = apply_random_displacements(atoms, sigma_per_species)
        
        # INVARIANT 1: compute nat and ntyp from the actual Atoms object
        nat = len(atoms)
        ntyp = len(set(atoms.get_chemical_symbols()))
        assert nat == 1, f"Expected nat=1 for Al primitive cell, got {nat}"
        assert ntyp == 1, f"Expected ntyp=1 for pure Al, got {ntyp}"
        
        # Write CIF
        write(out_path, atoms)
        
        if n == 0 or n % 10 == 0:
            print(f"  Wrote {out_path} (nat={nat}, ntyp={ntyp})")
    
    print(f"\nDone. Wrote {N_STRUCTURES} structures to {OUT_DIR}/")
    print(f"Temperature: {T_K} K")
    print(f"Al displacement: U = {U_Al_A2} A^2, sigma = {sigma_Al_A:.4f} A (per-axis RMS)")
    print(f"Lattice strain magnitude: {strain_mag*100:.1f}%")


if __name__ == '__main__':
    main()
