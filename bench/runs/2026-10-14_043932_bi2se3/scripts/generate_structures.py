"""
Generate 51 perturbed Bi2Se3 structures (structure_000.cif through structure_050.cif).

Structure 000 is the unperturbed primitive cell.
Structures 001-050 have:
  - Random atomic displacements (Einstein model, per-species amplitude)
  - Small random lattice strain (~0.5%)

Temperature: 300 K
  Bi2Se3 is the stable ambient-pressure phase at room temperature.
  Melting point ~1000 K, no structural phase transition below that.
  300 K is well within the stable range.

Displacement model: Einstein model
  U (mean-square displacement, A^2) = 3*hbar^2 / (4*pi^2 * m * k_B * T)
  Per-axis RMS displacement = sqrt(U) in A
  Lighter atoms (Se, Z=34) move more than heavier atoms (Bi, Z=83).

Strain: uniform random strain tensor, magnitude ~0.5% (typical thermal strain scale).
"""

import numpy as np
import ase.io
from ase import Atoms

# Physical constants in SI units (from scipy.constants, CODATA)
# scipy.constants uses 'Boltzmann' for k_B and 'hbar' for reduced Planck constant
import scipy.constants as const

T = 300.0  # K

# Atomic masses in kg
masses = {
    'Bi': 208.9804 * 1e-27,  # kg
    'Se': 78.9718 * 1e-27,   # kg
}

# Einstein model: U = 3*hbar^2 / (4*pi^2 * m * k_B * T)  [m^2]
# Convert to A^2: 1 m^2 = 1e20 A^2
U_A2 = {}
for el, m in masses.items():
    U_m2 = 3.0 * const.hbar**2 / (4.0 * np.pi**2 * m * const.Boltzmann * T)
    U_A2[el] = U_m2 * 1e20  # A^2
    print(f"{el}: U = {U_A2[el]:.6f} A^2, RMS = {np.sqrt(U_A2[el]):.6f} A")

# Verify: Se should have larger U than Bi (lighter atom)
assert U_A2['Se'] > U_A2['Bi'], "Se should have larger displacement than Bi"

# Strain magnitude: 0.5% = 0.005
strain_mag = 0.005

# Read the base structure
base = ase.io.read('structures/structure_000.cif')
print(f"Base structure: {len(base)} atoms, {len(set(base.get_chemical_symbols()))} species")
print(f"Base cell:\n{base.get_cell()}")

# INVARIANT 1: nat/ntyp from actual Atoms object
base_nat = len(base)
base_ntyp = len(set(base.get_chemical_symbols()))
assert base_nat == 5, f"Expected 5 atoms, got {base_nat}"
assert base_ntyp == 2, f"Expected 2 species, got {base_ntyp}"

rng = np.random.default_rng(42)  # fixed seed for reproducibility

for i in range(51):
    nnn = f"{i:03d}"
    
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()
        
        # Apply random atomic displacements (Einstein model)
        # Per-axis RMS displacement = sqrt(U) in A
        # Displacements are Gaussian with std = sqrt(U) per axis
        symbols = atoms.get_chemical_symbols()
        disp = np.zeros((len(atoms), 3))
        for j, sym in enumerate(symbols):
            rms = np.sqrt(U_A2[sym])  # A
            disp[j] = rng.normal(0.0, rms, size=3)
        
        # Apply displacements in Cartesian coordinates
        # Convert to Cartesian, add displacement, convert back to fractional
        cart = atoms.get_positions()  # Cartesian, A
        cart += disp
        atoms.set_positions(cart, apply_constraint=False)
        
        # Apply small random lattice strain
        # Random strain tensor: symmetric, traceless-ish, magnitude ~0.5%
        strain = rng.normal(0.0, strain_mag / np.sqrt(6), size=(3, 3))
        strain = (strain + strain.T) / 2.0  # make symmetric
        # Remove trace (volume-preserving) to keep it a pure shape distortion
        strain -= np.trace(strain) / 3.0 * np.eye(3)
        
        cell = atoms.get_cell()
        new_cell = cell + cell @ strain  # strain applied to cell vectors
        atoms.set_cell(new_cell, scale_atoms=True)  # scale atoms with cell
    
    # INVARIANT 1: assert nat/ntyp per structure
    assert len(atoms) == base_nat, f"Structure {nnn}: nat changed"
    assert len(set(atoms.get_chemical_symbols())) == base_ntyp, f"Structure {nnn}: ntyp changed"
    
    out_path = f'structures/structure_{nnn}.cif'
    ase.io.write(out_path, atoms)

print(f"\nGenerated 51 structures: structure_000.cif through structure_050.cif")
print(f"Temperature: {T} K")
print(f"Displacements (Einstein model, per-axis RMS in A):")
for el in ['Bi', 'Se']:
    print(f"  {el}: U = {U_A2[el]:.6f} A^2, RMS = {np.sqrt(U_A2[el]):.6f} A")
print(f"Strain magnitude: {strain_mag*100:.1f}%")
