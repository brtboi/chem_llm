"""
Generate 51 perturbed diamond-Si structures from the primitive cell.

Structure 000: unperturbed base (primitive cell from Materials Project mp-149).
Structures 001-050: random thermal displacements + small random lattice strain.

Temperature choice: 300 K (room temperature).
  - Diamond Si is the stable phase from 0 K up to the melting point (~1687 K).
  - 300 K is a representative condition where the phase is well-characterized
    and thermal displacements are small enough to preserve the diamond structure.

Displacement model (Einstein model, per-axis RMS):
  - For a harmonic oscillator at temperature T, the mean-square displacement is:
      <u^2> = 3 * k_B * T / (M * omega^2)
    where omega is the Einstein frequency.
  - For Si, the Debye temperature is ~645 K, and the Einstein frequency is
    approximately omega_E = sqrt(3) * omega_D (for a 3D solid).
    omega_D = k_B * T_D / h = 1.38e-23 * 645 / 6.626e-34 = 1.34e13 rad/s
    omega_E = sqrt(3) * 1.34e13 = 2.32e13 rad/s
  - Per-axis RMS displacement: sqrt(<u^2>/3) = sqrt(k_B * T / (M * omega_E^2))
    M = 28.0855 g/mol = 28.0855e-3 / 6.022e23 kg = 4.664e-26 kg
    k_B * T = 1.38e-23 * 300 = 4.14e-21 J
    M * omega_E^2 = 4.664e-26 * (2.32e13)^2 = 4.664e-26 * 5.38e26 = 25.1 kg/s^2
    <u^2>/3 = 4.14e-21 / 25.1 = 1.65e-22 m^2 = 1.65e-14 cm^2 = 1.65e-6 A^2
    RMS per axis = sqrt(1.65e-6) A = 0.00128 A

  This is very small. In practice, for DFT perturbation studies, we use a
  slightly larger amplitude to ensure the perturbation is numerically significant
  while still being physically reasonable. We use:
    RMS per axis = 0.05 A  (i.e., U = 0.0025 A^2 per axis)
  This is ~40x the Einstein estimate, which is acceptable for a perturbation
  study where we want to sample a range of local environments.

  Since Si is the only species, all atoms get the same amplitude.

Lattice strain: small random isotropic + shear strain, magnitude ~0.5%.
"""

import numpy as np
import ase.io
from ase import Atoms

# Read the primitive cell
base = ase.io.read('structures/prim_Si.cif')
print(f"Base structure: {len(base)} atoms, cell={base.cell}")

# Verify nat/ntyp invariant for the base structure
nat_base = len(base)
types_base = set(base.get_chemical_symbols())
types_base = sorted(types_base)
types_base = sorted(types_base)
ntyp_base = len(types_base)
print(f"Base: nat={nat_base}, ntyp={ntyp_base}, types={types_base}")
assert nat_base == 2, f"Expected 2 atoms in primitive Si cell, got {nat_base}"
assert ntyp_base == 1, f"Expected 1 species, got {ntyp_base}"

# Parameters
T = 300.0  # K
rms_per_axis = 0.05  # A (RMS displacement per axis, see docstring for justification)
strain_magnitude = 0.005  # 0.5% strain

np.random.seed(42)  # reproducibility

for i in range(51):
    nnn = f"{i:03d}"
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()

        # Apply small random lattice strain
        # Strain tensor: isotropic + small shear
        # epsilon = strain_magnitude * (random symmetric matrix with zero trace for shear)
        # For simplicity: small random strain tensor with components ~ strain_magnitude
        strain = np.random.normal(0.0, strain_magnitude, size=(3, 3))
        # Make it symmetric
        strain = 0.5 * (strain + strain.T)
        # Apply to cell: new_cell = (I + strain) @ old_cell
        # In ASE, cell vectors are rows, so: new_cell = old_cell @ (I + strain).T
        new_cell = atoms.cell @ (np.eye(3) + strain).T
        atoms.cell = new_cell
        # Update positions to maintain fractional coordinates
        # Actually, ASE handles this: when you set cell, positions are in Cartesian.
        # We need to transform positions too.
        # The strain is applied to the cell, so positions should be transformed:
        # new_pos = old_pos @ (I + strain).T  (since positions are in Cartesian)
        # But we already added displacements in the old Cartesian frame.
        # Let's redo: apply strain to both cell and positions.
        # Actually, the standard way: strain the cell, then the fractional
        # coordinates stay the same, so Cartesian positions change.
        # Let's do it properly:
        atoms2 = base.copy()
        # Apply strain to cell
        atoms2.cell = base.cell @ (np.eye(3) + strain).T
        # Fractional coordinates unchanged, so Cartesian positions are:
        # new_cart = frac @ new_cell
        frac = base.get_scaled_positions()
        atoms2.positions = frac @ atoms2.cell
        # Now add displacements in the new Cartesian frame
        n_atoms = len(atoms2)
        displacements = np.random.normal(0.0, rms_per_axis, size=(n_atoms, 3))
        atoms2.positions += displacements
        atoms = atoms2

    # Verify nat/ntyp invariant for each structure
    nat = len(atoms)
    types = set(atoms.get_chemical_symbols())
    ntyp = len(types)
    assert nat == nat_base, f"Structure {nnn}: nat changed from {nat_base} to {nat}"
    assert ntyp == ntyp_base, f"Structure {nnn}: ntyp changed from {ntyp_base} to {ntyp}"

    out_path = f"structures/structure_{nnn}.cif"
    ase.io.write(out_path, atoms)
    print(f"Wrote {out_path}: nat={nat}, ntyp={ntyp}")

print("Done: 51 structures written.")
