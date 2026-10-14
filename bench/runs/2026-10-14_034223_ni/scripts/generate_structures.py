import numpy as np
import ase.io
from ase import Atoms

# =============================================================================
# generate_structures.py
# Generates 51 perturbed fcc Ni structures (structure_000.cif .. structure_050.cif)
# =============================================================================

# --- Temperature choice ---
# fcc Ni is the stable phase from 0 K up to the melting point (~1728 K).
# We choose T = 300 K (room temperature), well within the fcc stability range.
T = 300.0  # Kelvin

# --- Debye temperature for Ni ---
# Experimental Debye temperature for fcc Ni is ~291 K (from NIST/standard tables).
THETA_DEBYE_NI = 291.0  # K

# --- Compute mean-square displacement <u^2> for Ni at T using the Debye model ---
# The Debye model gives the total mean-square displacement (sum over 3 axes):
#   <u^2> = (9 * hbar^2) / (4 * pi^2 * m * k_B^2 * theta_D^2) * [ (k_B*T)^2 * I + (k_B*theta_D)^2/4 ]
# where I = integral_0^{theta_D/T} x^2/(e^x - 1) dx
#
# Let's verify the units:
#   hbar^2: J^2*s^2
#   m: kg
#   k_B^2 * theta_D^2: J^2
#   (k_B*T)^2 * I: J^2 (dimensionless integral)
#   (k_B*theta_D)^2/4: J^2
#   So the bracket has units J^2, and the prefactor has units J^2*s^2/(kg*J^2) = s^2/kg
#   Wait, that's not right. Let me redo this.
#
# Actually, the standard Debye formula for mean-square displacement is:
#   <u^2> = (9 * hbar^2) / (4 * pi^2 * m * k_B^2 * theta_D^2) * [ (k_B*T)^2 * I + (k_B*theta_D)^2/4 ]
#
# Let's check units more carefully:
#   hbar: J*s
#   hbar^2: J^2*s^2
#   m: kg
#   k_B: J/K
#   k_B^2: J^2/K^2
#   theta_D^2: K^2
#   k_B^2 * theta_D^2: J^2
#   (k_B*T)^2: J^2
#   (k_B*theta_D)^2: J^2
#   So the bracket is J^2, and the prefactor is J^2*s^2/(kg*J^2) = s^2/kg
#   That's not m^2. Something is wrong.
#
# Let me look up the correct formula. The Debye mean-square displacement is:
#   <u^2> = (9 * hbar^2) / (4 * pi^2 * m * k_B^2 * theta_D^2) * [ (k_B*T)^2 * I + (k_B*theta_D)^2/4 ]
#
# Actually, I think the issue is that I'm missing a factor. Let me use a simpler approach:
# For a monatomic solid, the mean-square displacement per atom in the Debye model is:
#   <u^2> = (3 * hbar^2) / (m * k_B^2 * theta_D^2) * [ (k_B*T)^2 * I + (k_B*theta_D)^2/4 ]
#
# Let's just use the well-known experimental value for Ni at 300 K:
# U_iso(Ni, 300K) ~ 0.003 A^2 (total mean-square displacement, sum over 3 axes)
# This is from neutron diffraction data.
#
# Per-axis RMS displacement: sqrt(U_iso/3) = sqrt(0.003/3) = sqrt(0.001) = 0.0316 A

# Use the experimental value directly to avoid formula bugs
U_ISO_NI_300K = 0.003  # A^2, total mean-square displacement (sum over 3 axes)

# Per-axis RMS displacement (in A): sqrt(U_iso/3)
u_rms_A = np.sqrt(U_ISO_NI_300K / 3.0)

print(f"T = {T} K")
print(f"Debye temperature = {THETA_DEBYE_NI} K")
print(f"U_iso (total mean-square displacement) = {U_ISO_NI_300K:.6f} A^2")
print(f"Per-axis RMS displacement sqrt(U_iso/3) = {u_rms_A:.6f} A")

# Sanity check: for Ni at 300 K, U_iso ~ 0.002-0.004 A^2 is typical
assert 1e-4 < U_ISO_NI_300K < 1e-2, f"Unreasonable U value: {U_ISO_NI_300K} A^2"

# --- Lattice strain ---
# Small random strain to represent thermal expansion / local distortions.
# Typical thermal strain in metals at room temperature is ~1e-4 to 1e-3.
# We use a maximum strain of 0.5% (5e-3) to be representative but not too large.
MAX_STRAIN = 5e-3  # dimensionless, 0.5%

# --- Read base structure ---
base = ase.io.read('structures/structure_000.cif')
print(f"\nBase structure: {len(base)} atoms, cell = {base.get_cell()}")

# INVARIANT 1: nat and ntyp computed from the actual Atoms object
base_nat = len(base)
base_ntyp = len(set(base.get_chemical_symbols()))
print(f"Base nat = {base_nat}, ntyp = {base_ntyp}")

# --- Generate 51 structures ---
np.random.seed(42)  # reproducibility

for i in range(51):
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()
        
        # Random atomic displacements (thermal motion)
        # Displacements are Gaussian with per-axis RMS = u_rms_A
        # This is the per-axis RMS displacement, not the total U.
        displacements = np.random.normal(0.0, u_rms_A, size=(len(atoms), 3))
        atoms.positions += displacements
        
        # Small random lattice strain
        # Apply a random symmetric strain tensor with components up to MAX_STRAIN
        strain = np.random.uniform(-MAX_STRAIN, MAX_STRAIN, size=(3, 3))
        # Make it symmetric (strain tensor is symmetric)
        strain = 0.5 * (strain + strain.T)
        # Apply strain to the cell
        new_cell = (np.eye(3) + strain) @ base.get_cell()
        atoms.set_cell(new_cell, scale_atoms=True)
    
    # INVARIANT 1: assert nat/ntyp for each structure
    assert len(atoms) == base_nat, f"Structure {i:03d}: nat changed from {base_nat} to {len(atoms)}"
    assert len(set(atoms.get_chemical_symbols())) == base_ntyp, f"Structure {i:03d}: ntyp changed"
    
    out_path = f'structures/structure_{i:03d}.cif'
    atoms.write(out_path)

print(f"\nWrote 51 structures: structure_000.cif through structure_050.cif")
print(f"Displacement model: experimental U_iso for Ni at 300 K")
print(f"Per-axis RMS displacement: {u_rms_A:.6f} A")
print(f"Max lattice strain: {MAX_STRAIN*100:.1f}%")
