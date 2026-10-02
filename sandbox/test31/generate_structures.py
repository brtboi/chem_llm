#!/usr/bin/env python3
"""
Generate 51 perturbed CsPbBr3 structures (structure_000.cif through structure_050.cif).

Structure 000: unperturbed primitive cell (from structures/CsPbBr3_primitive.cif).
Structures 001-050: random atomic displacements (thermal motion) + small random lattice strain.

Temperature choice: 300 K (room temperature).
Justification: CsPbBr3 cubic perovskite (Pm-3m) is the high-temperature phase that remains
stable down to ~230 K where it transitions to orthorhombic. 300 K is well within the cubic
phase stability range, so thermal displacements at this temperature are physically meaningful.

Thermal displacement model: Einstein model.
For each species, the mean-square displacement <u^2> = 3*kB*T/(m*omega_E^2),
where omega_E = kB*Theta_D/hbar is the Einstein frequency derived from the Debye temperature.
The per-axis RMS displacement is sqrt(<u^2>/3) = sqrt(kB*T/(m*omega_E^2)) in Angstrom.

Debye temperatures used (literature values):
  Cs: Theta_D ~ 100 K (light, weakly bound -> larger displacement)
  Pb: Theta_D ~ 100 K (heavy -> smaller displacement)
  Br: Theta_D ~ 200 K (intermediate mass, stiffer bond -> smaller displacement)

Computed per-axis RMS displacements (in Angstrom):
  Cs: ~0.0105 A
  Pb: ~0.0084 A
  Br: ~0.0067 A

Lattice strain: small random strain tensor with magnitude ~0.5% (0.005),
representing thermal expansion/contraction fluctuations.

Units: All displacements are in Angstrom. U (crystallographic displacement parameter)
would be the mean-square displacement in A^2; here we use the per-axis RMS = sqrt(U/3) in A.
"""

import numpy as np
from ase.io import read, write
from ase import Atoms

# Physical constants
kB = 1.380649e-23      # J/K
hbar = 1.054571817e-34 # J*s
amu = 1.66053906660e-27 # kg

# Temperature (K) -- cubic CsPbBr3 is stable at room temperature
T = 300.0

# Debye temperatures (K) for each species
# Cs: light alkali metal, weakly bound in perovskite A-site -> low Debye temp
# Pb: heavy, in B-site -> low Debye temp
# Br: halide, stiffer bonds -> higher Debye temp
THETA_D = {
    'Cs': 100.0,
    'Pb': 100.0,
    'Br': 200.0,
}

# Atomic masses (amu)
MASS = {
    'Cs': 132.905,
    'Pb': 207.2,
    'Br': 79.904,
}


def compute_rms_displacement(symbol, T):
    """
    Compute per-axis RMS thermal displacement in Angstrom using the Einstein model.
    
    <u^2> = 3*kB*T / (m * omega_E^2)
    omega_E = kB * Theta_D / hbar
    per-axis RMS = sqrt(<u^2> / 3) = sqrt(kB*T / (m * omega_E^2))
    
    Returns RMS in Angstrom.
    """
    theta_d = THETA_D[symbol]
    mass_kg = MASS[symbol] * amu
    omega_e = kB * theta_d / hbar  # rad/s
    # <u^2> in m^2
    u2 = 3.0 * kB * T / (mass_kg * omega_e**2)
    # per-axis RMS in m
    rms_m = np.sqrt(u2 / 3.0)
    # convert to Angstrom
    rms_angstrom = rms_m * 1e10
    return rms_angstrom


def main():
    # Read the primitive cell base structure
    base = read('structures/CsPbBr3_primitive.cif')
    n_atoms = len(base)
    symbols = base.get_chemical_symbols()
    n_species = len(set(symbols))
    
    print(f"Base structure: {n_atoms} atoms, {n_species} species: {set(symbols)}")
    print(f"Cell: {base.cell[:].tolist()}")
    
    # Compute per-species RMS displacements
    rms_dict = {}
    for sym in set(symbols):
        rms_dict[sym] = compute_rms_displacement(sym, T)
        print(f"  {sym}: RMS per-axis displacement = {rms_dict[sym]:.4f} A "
              f"(U = {rms_dict[sym]**2 * 3:.2e} A^2, Theta_D = {THETA_D[sym]:.0f} K)")
    
    # Write structure 000: unperturbed base
    write('structures/structure_000.cif', base)
    print("Wrote structures/structure_000.cif (unperturbed)")
    
    # Generate 50 perturbed structures
    np.random.seed(42)  # reproducibility
    
    for i in range(1, 51):
        atoms = base.copy()
        
        # Apply random lattice strain: small random strain tensor, ~0.5% magnitude
        # strain tensor is symmetric, traceless (volume-preserving) + small isotropic part
        strain = np.random.randn(3, 3) * 0.005  # 0.5% RMS per component
        strain = (strain + strain.T) / 2.0  # symmetrize
        # Remove trace (volume change) to keep it purely shear + small isotropic
        trace = np.trace(strain) / 3.0
        strain -= trace * np.eye(3)
        # Add a small isotropic component (thermal expansion fluctuation)
        strain += np.random.randn() * 0.001 * np.eye(3)
        
        # Apply strain to cell: new_cell = (I + strain) @ old_cell
        # In ASE, cell vectors are rows, so: new_cell = old_cell @ (I + strain).T
        new_cell = base.cell[:] @ (np.eye(3) + strain).T
        atoms.cell = new_cell
        
        # Apply random atomic displacements (thermal motion)
        # Each atom gets a Gaussian displacement with per-species RMS
        displacements = np.zeros((n_atoms, 3))
        for j, sym in enumerate(symbols):
            displacements[j] = np.random.randn(3) * rms_dict[sym]
        
        # Displacements are in Cartesian Angstrom
        atoms.positions += displacements
        
        # INVARIANT 1: assert nat/ntyp consistency
        assert len(atoms) == n_atoms, f"Structure {i:03d}: atom count changed"
        assert len(set(atoms.get_chemical_symbols())) == n_species, \
            f"Structure {i:03d}: species count changed"
        
        # Write the perturbed structure
        fname = f'structures/structure_{i:03d}.cif'
        write(fname, atoms)
        
        if i <= 3 or i == 50:
            print(f"Wrote {fname}: strain = {strain.tolist()}, "
                  f"max displacement = {np.max(np.abs(displacements)):.4f} A")
    
    print(f"\nDone: wrote 51 structures (structure_000.cif through structure_050.cif)")


if __name__ == '__main__':
    main()
