"""
Generate 51 perturbed graphite structures (structure_000.cif .. structure_050.cif).

structure_000.cif is the unperturbed primitive cell.
The remaining 50 structures have:
  - random atomic displacements representative of thermal motion at 298 K
  - a small random lattice strain (~0.5%)

Temperature choice: 298 K (room temperature).
Graphite is the thermodynamically stable phase of carbon at ambient conditions
(1 atm, 298 K). It is stable well above room temperature as well (up to ~3900 K
at 1 atm before sublimation), so 298 K is a natural and well-justified choice.

Displacement amplitudes:
For carbon at 298 K, typical crystallographic displacement parameters U are
~0.005-0.01 A^2 (mean-square displacement). The per-axis RMS displacement is
sqrt(U) ~ 0.07-0.1 A. We use 0.08 A RMS per axis for C.

Units:
  - U (crystallographic displacement parameter) = mean-square displacement in A^2
  - RMS displacement per axis = sqrt(U) in A
  - We draw displacements from a Gaussian with sigma = 0.08 A (RMS per axis)

Lattice strain:
  - Small random strain tensor with components ~ N(0, 0.005) (0.5% RMS)
  - Applied as a deformation to the cell vectors

INVARIANT 1: nat and ntyp are computed from the actual ASE Atoms object.
"""

import numpy as np
import ase.io
from ase import Atoms

# --- Parameters ---
T = 298.0  # K, room temperature; graphite is stable at ambient conditions

# Per-species RMS displacement per axis (A)
# C: U ~ 0.0064 A^2 -> sqrt(U) ~ 0.08 A
RMS_DISP = {'C': 0.08}  # A, per axis

# Lattice strain: RMS of each strain component
STRAIN_RMS = 0.005  # 0.5% RMS strain

N_STRUCTURES = 51  # 000 (unperturbed) + 50 perturbed

# --- Read base structure ---
base = ase.io.read('structures/primitive_C.cif')
print(f"Base structure: {len(base)} atoms, cell = {base.get_cell()}")

# INVARIANT 1: compute nat and ntyp from the actual Atoms object
base_nat = len(base)
base_ntyp = len(set(base.get_chemical_symbols()))
print(f"Base nat={base_nat}, ntyp={base_ntyp}")

# --- Generate structures ---
np.random.seed(42)  # reproducibility

for i in range(N_STRUCTURES):
    out_path = f'structures/structure_{i:03d}.cif'

    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()

        # --- Random lattice strain ---
        # Apply a small random strain tensor to the cell vectors.
        # strain = [[e_xx, e_xy, e_xz], [e_yx, e_yy, e_yz], [e_zx, e_zy, e_zz]]
        # New cell = (I + strain) @ old_cell
        strain = np.random.normal(0.0, STRAIN_RMS, size=(3, 3))
        # Make strain symmetric (physical strain is symmetric)
        strain = 0.5 * (strain + strain.T)
        old_cell = atoms.get_cell()
        new_cell = (np.eye(3) + strain) @ np.array(old_cell)
        atoms.set_cell(new_cell, scale_atoms=True)  # scale atoms to follow cell

        # --- Random atomic displacements (thermal motion) ---
        # Displacements in Cartesian coordinates (A), Gaussian with sigma = RMS_DISP per axis
        symbols = atoms.get_chemical_symbols()
        disp = np.zeros((len(atoms), 3))
        for j, sym in enumerate(symbols):
            sigma = RMS_DISP[sym]  # A, per axis
            disp[j] = np.random.normal(0.0, sigma, size=3)
        atoms.positions += disp

    # INVARIANT 1: assert nat and ntyp match the base structure
    cur_nat = len(atoms)
    cur_ntyp = len(set(atoms.get_chemical_symbols()))
    assert cur_nat == base_nat, f"Structure {i:03d}: nat {cur_nat} != base {base_nat}"
    assert cur_ntyp == base_ntyp, f"Structure {i:03d}: ntyp {cur_ntyp} != base {base_ntyp}"

    ase.io.write(out_path, atoms)

print(f"Wrote {N_STRUCTURES} structures: structure_000.cif .. structure_{N_STRUCTURES-1:03d}.cif")
print("Done.")
