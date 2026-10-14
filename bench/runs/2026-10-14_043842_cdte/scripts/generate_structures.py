import numpy as np
import ase.io
import os

# Read the base (unperturbed) structure
base = ase.io.read('structures/structure_000.cif')
print(f"Base structure: {len(base)} atoms, symbols: {base.get_chemical_symbols()}")

# Temperature choice: 300 K
# CdTe is stable at room temperature (melting point ~1045 K), so 300 K is representative.
# Einstein model for thermal displacements:
#   U (mean-square displacement in A^2) = 3*k_B*T / (4*pi^2 * m * nu^2)
#   RMS displacement = sqrt(U) in A
# Using Debye temperatures: Cd ~150 K, Te ~164 K
# At 300 K: k_B*T = 0.02585 eV = 4.14e-21 J
# For Cd (mass 112.4 amu): RMS ~ 0.05 A
# For Te (mass 127.6 amu): RMS ~ 0.04 A
# These are RMS displacements (sqrt(U)) in Angstroms.

T = 300.0  # K
k_B_J = 1.380649e-23  # J/K (Boltzmann constant in SI units)

# Characteristic frequencies from Debye temperatures
# nu = k_B * T_D / h  (in Hz)
h = 6.62607015e-34  # J*s (Planck constant in SI units)
T_D_Cd = 150.0  # K
T_D_Te = 164.0  # K
nu_Cd = k_B_J * T_D_Cd / h  # Hz
nu_Te = k_B_J * T_D_Te / h  # Hz

# Einstein model: U = 3*k_B*T / (4*pi^2 * m * nu^2)
# m in kg, nu in Hz, U in m^2 -> convert to A^2
amu_to_kg = 1.66053906660e-27
m_Cd = 112.4 * amu_to_kg
m_Te = 127.6 * amu_to_kg

U_Cd = 3 * k_B_J * T / (4 * np.pi**2 * m_Cd * nu_Cd**2)  # m^2
U_Te = 3 * k_B_J * T / (4 * np.pi**2 * m_Te * nu_Te**2)  # m^2

# Convert to A^2 (1 m = 1e10 A)
U_Cd_A2 = U_Cd * 1e20  # A^2
U_Te_A2 = U_Te * 1e20  # A^2

rms_Cd = np.sqrt(U_Cd_A2)  # A
rms_Te = np.sqrt(U_Te_A2)  # A

print(f"Cd: U = {U_Cd_A2:.4f} A^2, RMS = {rms_Cd:.4f} A")
print(f"Te: U = {U_Te_A2:.4f} A^2, RMS = {rms_Te:.4f} A")

# Sanity check: RMS displacements should be small (less than 1 A)
assert rms_Cd < 1.0, f"Cd RMS displacement too large: {rms_Cd} A"
assert rms_Te < 1.0, f"Te RMS displacement too large: {rms_Te} A"

# Small random lattice strain: ~0.5% max
max_strain = 0.005

rng = np.random.default_rng(42)

for i in range(51):
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()
        
        # Random atomic displacements (thermal motion)
        for idx, atom in enumerate(atoms):
            symbol = atom.symbol
            if symbol == 'Cd':
                rms = rms_Cd
            elif symbol == 'Te':
                rms = rms_Te
            else:
                rms = 0.05  # default
            # Displacement in Cartesian coordinates (A)
            disp = rng.normal(0, rms, 3)
            atoms.positions[idx] += disp
        
        # Small random lattice strain
        strain = rng.uniform(-max_strain, max_strain, (3, 3))
        # Make strain symmetric (no rotation component)
        strain = 0.5 * (strain + strain.T)
        # Apply strain to the cell
        new_cell = (np.eye(3) + strain) @ atoms.get_cell().array
        atoms.set_cell(new_cell, scale_atoms=True)
    
    # INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Structure {i:03d}: expected 2 atoms, got {nat}"
    assert ntyp == 2, f"Structure {i:03d}: expected 2 species, got {ntyp}"
    
    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)

print(f"Wrote 51 structures: structure_000.cif through structure_050.cif")
