"""
Generate 51 perturbed graphite structures (structure_000.cif through structure_050.cif).

Structure 000: unperturbed base (primitive cell of mp-48, P6_3/mmc #194).
Structures 001-050: random atomic displacements + small random lattice strain.

Temperature choice: T = 300 K
  Graphite is the thermodynamically stable phase of carbon at ambient conditions
  (up to ~4000 K at 1 atm). 300 K is well within its stability range.

Thermal displacement amplitudes:
  For carbon at 300 K, the crystallographic mean-square displacement U is ~0.005 A^2.
  The per-axis RMS displacement is sqrt(U) = sqrt(0.005) ~ 0.071 A.
  Since there is only one species (C), all atoms get the same amplitude.
  We use RMS = 0.07 A per axis (i.e., U = 0.0049 A^2).
  Displacements are drawn from a Gaussian with std = 0.07 A along each Cartesian axis.

Lattice strain:
  Small random strain tensor with components ~ N(0, 0.005) (0.5% RMS),
  representing thermal expansion / anharmonic lattice effects.
  Applied as a deformation to the cell vectors.
"""

import numpy as np
import ase.io
from ase import Atoms
from chem_llm.kpath import standard_kpath, kp_atoms

# Read the base primitive cell
atoms_base = ase.io.read('structures/base_C_prim.cif')
print(f"Base structure: {len(atoms_base)} atoms, cell = {atoms_base.get_cell()}")

# Verify space group is preserved
kp = standard_kpath(atoms_base)
print(f"Space group: {kp['spacegroup']}")

# Number of structures to generate
N_STRUCTURES = 51

# Displacement parameters (see docstring for justification)
RMS_DISPLACEMENT = 0.07  # A, per-axis RMS (sqrt of U = 0.0049 A^2)
STRAIN_RMS = 0.005       # 0.5% RMS strain per component

# Random seed for reproducibility
rng = np.random.default_rng(seed=42)

for i in range(N_STRUCTURES):
    atoms = Atoms(
        symbols=atoms_base.get_chemical_symbols(),
        positions=atoms_base.get_positions().copy(),
        cell=atoms_base.get_cell().copy(),
        pbc=True,
    )

    if i == 0:
        # Unperturbed base structure
        pass
    else:
        # Random atomic displacements (Gaussian, per-axis std = RMS_DISPLACEMENT A)
        displacements = rng.normal(loc=0.0, scale=RMS_DISPLACEMENT, size=atoms.positions.shape)
        atoms.positions += displacements

        # Small random lattice strain: symmetric strain tensor
        # strain = [[exx, exy, exz], [exy, eyy, eyz], [exz, eyz, ezz]]
        exx = rng.normal(0, STRAIN_RMS)
        eyy = rng.normal(0, STRAIN_RMS)
        ezz = rng.normal(0, STRAIN_RMS)
        exy = rng.normal(0, STRAIN_RMS)
        exz = rng.normal(0, STRAIN_RMS)
        eyz = rng.normal(0, STRAIN_RMS)
        strain = np.array([
            [exx, exy, exz],
            [exy, eyy, eyz],
            [exz, eyz, ezz]
        ])
        # Apply strain: new_cell = old_cell @ (I + strain)^T
        # (strain is applied in Cartesian, so cell vectors transform as row vectors)
        deformation = np.eye(3) + strain
        atoms.cell = atoms.cell @ deformation.T

    # INVARIANT 1: nat and ntyp must be consistent with the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 4, f"Structure {i:03d}: expected nat=4, got {nat}"
    assert ntyp == 1, f"Structure {i:03d}: expected ntyp=1, got {ntyp}"

    # Write the structure
    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)

    if i == 0:
        print(f"Wrote {out_path} (unperturbed base)")
    else:
        print(f"Wrote {out_path} (perturbed: max |disp| = {np.max(np.abs(atoms.positions - atoms_base.positions)):.4f} A)")

print(f"\nDone: generated {N_STRUCTURES} structures (structure_000.cif to structure_050.cif)")
