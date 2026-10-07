import numpy as np
from ase.io import read, write
from ase import Atoms
import os

# =============================================================================
# generate_structures.py
# Generates 51 perturbed graphite structures (structure_000.cif through
# structure_050.cif) from the primitive cell saved in structures/primitive_C.cif.
# Structure 000 is the unperturbed base; 001-050 have random thermal
# displacements + small random lattice strain.
# =============================================================================

# --- Temperature choice ---
# Graphite (Bernal stacking, P6_3/mmc) is the thermodynamically stable phase
# of carbon at ambient pressure from 0 K up to the sublimation point (~3900 K).
# It is stable at room temperature (300 K). We choose T = 300 K as a
# representative temperature at which this phase is well-established and
# thermal displacements are physically meaningful but small.
T = 300.0  # Kelvin

# --- Per-species displacement amplitudes ---
# Only one species here: Carbon (Z=6).
# For a harmonic oscillator, the mean-square displacement is <u^2> = k_B*T / k,
# where k is the effective spring constant. For carbon in graphite, the
# in-plane C-C bond is strong (k ~ 15-20 N/m), while the out-of-plane
# interlayer interaction is much weaker (van der Waals, k ~ 0.1-1 N/m).
#
# We use a single isotropic RMS displacement per species as a simplification.
# A reasonable estimate for C at 300 K in graphite:
#   - In-plane: sqrt(k_B*T / k_inplane) ~ sqrt(4e-21 / 17) ~ 0.015 A
#   - Out-of-plane: sqrt(k_B*T / k_out) ~ sqrt(4e-21 / 0.5) ~ 0.09 A
#
# We take a representative RMS displacement of 0.03 A (30 pm) for carbon,
# which is between the in-plane and out-of-plane estimates and consistent
# with experimental Debye-Waller factors for graphite at 300 K
# (U_iso ~ 0.001-0.005 A^2, so sqrt(U) ~ 0.03-0.07 A).
#
# Units: RMS_C is the per-axis RMS displacement in Angstroms.
# The mean-square displacement U = RMS_C^2 is in A^2.
RMS_C = 0.03  # Angstroms, per-axis RMS displacement for Carbon at 300 K

# --- Lattice strain ---
# Small random strain to represent thermal expansion / anharmonic effects.
# Typical thermal strain in graphite at 300 K is ~1e-4 to 1e-3.
# We use a maximum strain of 0.001 (0.1%) per axis, applied as a random
# diagonal strain tensor (no shear, to preserve the hexagonal symmetry
# approximately).
MAX_STRAIN = 0.001  # dimensionless, max per-axis strain

# --- Number of structures ---
N_STRUCTURES = 51  # 000 (unperturbed) + 001-050 (perturbed)

# --- Read the primitive cell ---
base = read('structures/primitive_C.cif')
print(f"Base structure: {len(base)} atoms, species: {base.get_chemical_symbols()}")
print(f"Base cell:\n{base.cell}")

# Record nat/ntyp for the base (INVARIANT 1: compute from actual object)
nat_base = len(base)
types_base = set(base.get_chemical_symbols())
ntyp_base = len(types_base)
print(f"Base: nat={nat_base}, ntyp={ntyp_base}")

# --- Random seed for reproducibility ---
np.random.seed(42)

# --- Generate structures ---
for i in range(N_STRUCTURES):
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()

        # Apply small random lattice strain (diagonal, no shear)
        # strain_tensor is dimensionless; new_cell = old_cell @ (I + strain)
        strain = np.random.uniform(-MAX_STRAIN, MAX_STRAIN, size=3)
        strain_tensor = np.diag(strain)
        new_cell = atoms.cell @ (np.eye(3) + strain_tensor)
        atoms.cell = new_cell

        # Apply random thermal displacements to each atom
        # Displacements are in Cartesian coordinates (Angstroms)
        # Each atom gets a Gaussian displacement with std = RMS_C per axis
        displacements = np.random.normal(0.0, RMS_C, size=(len(atoms), 3))
        atoms.positions += displacements

    # INVARIANT 1: assert nat/ntyp are consistent with the actual object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == nat_base, f"Structure {i:03d}: nat changed from {nat_base} to {nat}"
    assert ntyp == ntyp_base, f"Structure {i:03d}: ntyp changed from {ntyp_base} to {ntyp}"

    # Write the structure
    out_path = f'structures/structure_{i:03d}.cif'
    write(out_path, atoms)
    print(f"Wrote {out_path}: nat={nat}, ntyp={ntyp}")

print(f"\nDone. Generated {N_STRUCTURES} structures in structures/")
