import numpy as np
import ase.io
from ase import Atoms

# ============================================================
# generate_structures.py
# Generates 51 Al structures: structure_000.cif (unperturbed)
# and structure_001.cif ... structure_050.cif (perturbed).
# ============================================================

# --- Temperature choice ---
# Al is fcc (Fm-3m) at all temperatures up to its melting point (933 K).
# We choose T = 300 K (room temperature), well within the stable fcc range.
# At 300 K the Debye-Waller factor for Al is small; displacements are
# representative of thermal motion at ambient conditions.
T = 300.0  # Kelvin

# --- Displacement amplitudes ---
# For a single-element system (Al), we use the Einstein model to estimate
# the mean-square displacement <u^2> = 3 k_B T / k, where k is the
# effective spring constant. For Al, the Debye temperature is ~428 K.
# At T = 300 K, the RMS displacement per axis is approximately:
#   u_rms = sqrt(3 * k_B * T / (m * omega^2))
# Using the Debye model, at T << T_D:
#   <u^2> = (9 * k_B * T / (4 * m * omega_D^2)) * (T_D/T)^4 * integral...
# A simpler estimate: for Al at 300 K, the mean-square displacement
# is about 0.005 A^2 (from experimental Debye-Waller factors).
# Per-axis RMS displacement = sqrt(U) = sqrt(0.005) ~ 0.07 A.
# We use U = 0.005 A^2 as the mean-square displacement per axis.
# This is the U parameter (crystallographic displacement parameter),
# which is the mean-square displacement in A^2.
# The per-axis RMS displacement is sqrt(U) = 0.0707 A.

U_Al = 0.005  # A^2, mean-square displacement per axis for Al at 300 K
# Per-axis RMS displacement: sqrt(U_Al) = 0.0707 A

# --- Lattice strain ---
# Small random lattice strain: each diagonal component of the strain
# tensor is drawn from a uniform distribution in [-0.01, 0.01] (1% strain).
# This represents small deviations from the ideal lattice constant.
strain_max = 0.01  # 1% maximum strain

# --- Read base structure ---
base = ase.io.read('structures/Al_primitive.cif')
print(f"Base structure: {len(base)} atoms, cell = {base.get_cell()}")

# --- Generate structures ---
np.random.seed(42)  # reproducibility

for i in range(51):
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()
        
        # Random atomic displacements (thermal motion)
        # Displacements are in Cartesian coordinates (Angstroms).
        # Each atom gets a random displacement drawn from a Gaussian
        # with standard deviation sqrt(U_Al) per axis.
        n_atoms = len(atoms)
        displacements = np.random.normal(0.0, np.sqrt(U_Al), size=(n_atoms, 3))
        atoms.positions += displacements
        
        # Small random lattice strain
        # Apply a random diagonal strain tensor to the cell.
        # strain[i] is the fractional change in the i-th lattice vector.
        strain = np.random.uniform(-strain_max, strain_max, size=3)
        # Apply strain: new_cell = old_cell * (1 + strain) along each axis
        cell = atoms.get_cell()
        new_cell = np.array([
            cell[0] * (1.0 + strain[0]),
            cell[1] * (1.0 + strain[1]),
            cell[2] * (1.0 + strain[2]),
        ])
        atoms.set_cell(new_cell, scale_atoms=True)
    
    # INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 1, f"Expected 1 atom, got {nat}"
    assert ntyp == 1, f"Expected 1 species, got {ntyp}"
    
    # Write structure
    fname = f'structures/structure_{i:03d}.cif'
    ase.io.write(fname, atoms)
    print(f"Wrote {fname}: nat={nat}, ntyp={ntyp}")

print("Done: 51 structures generated.")
