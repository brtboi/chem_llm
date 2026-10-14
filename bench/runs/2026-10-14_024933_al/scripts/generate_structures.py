import numpy as np
import ase.io
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

# =============================================================================
# generate_structures.py
# Generates 51 Al structures: structure_000.cif (unperturbed) through
# structure_050.cif (perturbed with thermal displacements + random strain).
# =============================================================================

# --- Temperature choice ---
# Al is fcc (Fm-3m) at all temperatures up to its melting point (933 K).
# We choose T = 300 K (room temperature), well within the stable fcc range.
# At 300 K the Debye-Waller factor for Al is small but non-zero.
# The Debye temperature of Al is ~428 K, so 300 K is in the classical-ish
# regime where equipartition gives a reasonable estimate of RMS displacement.

T = 300.0  # Kelvin

# --- Per-species RMS displacement amplitudes ---
# For a single-element system (Al), we use a single amplitude.
# The mean-square displacement U (in A^2) for an atom in a harmonic solid
# at temperature T can be estimated from the Debye model:
#   U = (9 * hbar^2 * T) / (4 * pi^2 * m * theta_D^2)  (high-T limit)
# For Al: m = 26.98 u = 4.48e-26 kg, theta_D = 428 K
#   U = 9 * (1.055e-34)^2 * 300 / (4 * pi^2 * 4.48e-26 * 428^2)
#   U = 9 * 1.113e-68 * 300 / (4 * 9.87 * 4.48e-26 * 183184)
#   U = 2.985e-65 / (8.09e-18) = 3.69e-48  ... this is in SI (m^2)
#   U = 3.69e-48 m^2 = 3.69e-8 A^2
# RMS displacement per axis = sqrt(U) = sqrt(3.69e-8) A = 1.92e-4 A
# This is very small. For a more practical perturbation that is still
# "representative of thermal motion" but large enough to be numerically
# meaningful in a DFT calculation, we use a slightly larger amplitude.
# In practice, for perturbation studies, amplitudes of ~0.01-0.05 A are
# commonly used to represent thermal motion without destroying the structure.
# We use RMS = 0.02 A per axis (U = 4e-4 A^2 per axis).
# This is ~100x the Debye-model value at 300 K, but is a standard choice
# for perturbation studies to ensure the perturbation is not lost in
# numerical noise while still being small compared to the lattice constant.

# Al is the only species. Lighter atoms move more; Al is the only atom here.
# U_Al = 4e-4 A^2 (mean-square displacement per axis)
# RMS_Al = sqrt(U_Al) = 0.02 A (RMS displacement per axis)

U_Al = 4e-4  # A^2, mean-square displacement per axis
RMS_Al = np.sqrt(U_Al)  # A, RMS displacement per axis = 0.02 A

# --- Random strain ---
# Small random lattice strain: each diagonal component of the strain tensor
# is drawn from a uniform distribution in [-0.01, 0.01] (i.e. up to 1% strain).
# This is small enough to keep the structure close to equilibrium.

strain_max = 0.01  # maximum strain per axis (1%)

# --- Read base structure ---
base = ase.io.read('structures/Al_primitive.cif')
print(f"Base structure: {len(base)} atoms, cell={base.cell[:].tolist()}")

# --- Generate 51 structures ---
np.random.seed(42)  # reproducibility

for i in range(51):
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()
        
        # Random atomic displacements (thermal motion)
        # Displacements are in Cartesian coordinates (A)
        # Each atom gets a random displacement drawn from a Gaussian
        # with standard deviation = RMS_Al per axis
        displacements = np.random.normal(0.0, RMS_Al, size=(len(atoms), 3))
        atoms.positions += displacements
        
        # Small random lattice strain
        # Apply a random diagonal strain tensor
        strain = np.random.uniform(-strain_max, strain_max, size=3)
        strain_tensor = np.diag(strain)
        # New cell = old cell * (I + strain_tensor)
        # In ASE, cell is a 3x3 matrix of vectors
        # Strain is applied as: new_cell = (I + strain) @ old_cell
        # But for diagonal strain, it's simpler to scale each axis
        new_cell = base.cell.copy()
        new_cell[0] *= (1.0 + strain[0])
        new_cell[1] *= (1.0 + strain[1])
        new_cell[2] *= (1.0 + strain[2])
        atoms.cell = new_cell
        # Rescale positions to maintain fractional coordinates
        # Actually, we should rescale positions to match the new cell
        # The positions were displaced in the old cell's Cartesian frame.
        # We need to transform them to the new cell.
        # Simplest: get fractional positions in old cell, apply to new cell
        frac = base.get_scaled_positions()
        # Add the displacement in fractional coordinates
        # displacement_fractional = displacement_cartesian @ inv(old_cell)
        old_cell_inv = np.linalg.inv(base.cell[:])
        disp_frac = displacements @ old_cell_inv
        frac_new = frac + disp_frac
        atoms.set_scaled_positions(frac_new)
    
    # INVARIANT 1: assert nat/ntyp consistency
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 1, f"Structure {i:03d}: expected 1 atom, got {nat}"
    assert ntyp == 1, f"Structure {i:03d}: expected 1 type, got {ntyp}"
    
    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)
    print(f"Wrote {out_path}: {nat} atoms, {ntyp} types")

print("Done: 51 structures written.")
