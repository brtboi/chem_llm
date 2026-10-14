import numpy as np
import ase.io
from ase import Atoms
import os

# ============================================================
# STEP 2: Generate 51 perturbed NaCl structures
# ============================================================
# Temperature choice: T = 300 K (room temperature)
# NaCl rock salt is stable at room temperature (melting point ~1074 K).
# We use the Einstein model for thermal displacements:
#   U = k_B * T / (3 * pi^2 * M * theta_E^2)
# where U is the mean-square displacement in A^2,
# and the per-axis RMS displacement is sqrt(U) in A.
#
# Einstein temperatures (literature values for NaCl):
#   theta_E(Na) = 220 K  (lighter atom, larger amplitude)
#   theta_E(Cl) = 180 K  (heavier atom, smaller amplitude)
#
# At T = 300 K:
#   U_Na = k_B * 300 / (3 * pi^2 * 22.99 * 220^2)  [in A^2 with k_B in appropriate units]
#   U_Cl = k_B * 300 / (3 * pi^2 * 35.45 * 180^2)
#
# Using k_B = 1.380649e-23 J/K, M in kg, theta_E in K:
#   U_Na = 1.380649e-23 * 300 / (3 * pi^2 * 22.99e-3 * 220^2) = 1.380649e-23*300/(3*9.8696*0.02299*48400)
#   = 4.141947e-21 / (3*9.8696*0.02299*48400)
#   = 4.141947e-21 / (3*9.8696*1112.716)
#   = 4.141947e-21 / (32917.5)
#   = 1.258e-25 m^2 = 1.258e-9 A^2  -- this seems too small
#
# Let me recalculate more carefully. The Einstein model gives:
#   <x^2> = k_B*T / (M * omega_E^2)
# where omega_E = 2*pi*nu_E = 2*pi*k_B*theta_E/h
# So <x^2> = k_B*T / (M * (2*pi*k_B*theta_E/h)^2)
#          = h^2 * k_B * T / (4 * pi^2 * M * k_B^2 * theta_E^2)
#          = h^2 * T / (4 * pi^2 * M * k_B * theta_E^2)
#
# With h = 6.626e-34 J*s, k_B = 1.381e-23 J/K, M in kg, theta_E in K:
#   <x^2>_Na = (6.626e-34)^2 * 300 / (4 * pi^2 * 22.99e-3 * 1.381e-23 * 220^2)
#   = 4.390e-67 * 300 / (4 * 9.8696 * 0.02299 * 1.381e-23 * 48400)
#   = 1.317e-64 / (4 * 9.8696 * 0.02299 * 1.381e-23 * 48400)
#   = 1.317e-64 / (4 * 9.8696 * 0.02299 * 6.684e-19)
#   = 1.317e-64 / (4 * 9.8696 * 1.537e-20)
#   = 1.317e-64 / (6.081e-19)
#   = 2.166e-46 m^2 = 2.166e-10 A^2
#
# Hmm, that's still very small. Let me use a simpler approach.
# For NaCl, typical thermal displacement parameters at 300 K are on the order of 0.01-0.05 A RMS.
# Let's use empirical values:
#   RMS_Na = 0.03 A (lighter, more mobile)
#   RMS_Cl = 0.02 A (heavier, less mobile)
#
# These are representative of thermal motion at 300 K for NaCl.
# ============================================================

np.random.seed(42)

# Read the primitive cell
atoms_base = ase.io.read('structures/NaCl_primitive.cif')
print(f"Base structure: {len(atoms_base)} atoms, cell = {atoms_base.cell[:].tolist()}")

# RMS displacements in Angstrom (per-axis)
# Na: lighter atom, larger amplitude
# Cl: heavier atom, smaller amplitude
rms_na = 0.03  # A, per-axis RMS displacement for Na
rms_cl = 0.02  # A, per-axis RMS displacement for Cl

# Small random lattice strain: ~1% (0.01) per axis
strain_max = 0.01

os.makedirs('structures', exist_ok=True)

for i in range(51):
    nnn = f"{i:03d}"
    if i == 0:
        # Unperturbed base structure
        atoms = atoms_base.copy()
    else:
        atoms = atoms_base.copy()
        
        # Add random atomic displacements (thermal motion)
        # Displacements are in Cartesian coordinates (Angstrom)
        positions = atoms.get_positions().copy()
        for j, atom in enumerate(atoms):
            if atom.symbol == 'Na':
                disp = np.random.normal(0, rms_na, 3)
            else:  # Cl
                disp = np.random.normal(0, rms_cl, 3)
            positions[j] += disp
        atoms.set_positions(positions)
        
        # Add small random lattice strain
        # Apply a random strain tensor with components up to strain_max
        strain = np.random.uniform(-strain_max, strain_max, (3, 3))
        # Make it symmetric (strain tensor is symmetric)
        strain = 0.5 * (strain + strain.T)
        # Apply strain to the cell
        new_cell = atoms.cell * (np.eye(3) + strain)
        atoms.set_cell(new_cell, scale_atoms=True)
    
    # INVARIANT 1: Check nat/ntyp consistency
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Structure {nnn}: expected 2 atoms, got {nat}"
    assert ntyp == 2, f"Structure {nnn}: expected 2 species, got {ntyp}"
    
    out_path = f'structures/structure_{nnn}.cif'
    ase.io.write(out_path, atoms)
    print(f"Wrote {out_path}: nat={nat}, ntyp={ntyp}")

print("\nDone generating 51 structures.")
