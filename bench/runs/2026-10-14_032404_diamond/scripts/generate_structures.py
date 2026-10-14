import numpy as np
import ase.io

# =============================================================================
# generate_structures.py
# Generates 51 diamond-C structures: structure_000.cif (unperturbed) through
# structure_050.cif (perturbed with thermal displacements + random strain).
# =============================================================================

# --- Temperature choice ---
# Diamond is the stable phase of carbon at ambient pressure up to its
# sublimation temperature (~4000 K).  We choose T = 1000 K, well within the
# stability range, and high enough to give physically meaningful thermal
# displacements (RMS ~0.05 A for C).
T = 1000.0  # Kelvin

# --- Einstein-model displacement amplitudes ---
# For a monatomic crystal (diamond C) the Einstein model gives:
#   <u^2> = 3 k_B T / k_spring
# where k_spring is the effective spring constant.
# For diamond, the Debye temperature is ~1800 K, so at T = 1000 K the
# mean-square displacement per atom is approximately:
#   <u^2> ~ (hbar^2 / (2 m k_B T)) * (T / theta_D)^3  (high-T limit)
# A simpler estimate: at T = 1000 K, the RMS displacement of C in diamond
# is about 0.05 A (from neutron diffraction data).
# We use RMS = 0.05 A per axis, i.e. U = 0.0025 A^2 (mean-square displacement).
# U is the crystallographic displacement parameter in A^2.
# Per-axis RMS = sqrt(U) = 0.05 A.

# Only one species: C
# Amplitude (RMS per axis) in Angstrom:
RMS_C = 0.05  # A  (per-axis RMS displacement at T = 1000 K)
U_C = RMS_C**2  # A^2 (mean-square displacement, crystallographic U)

# --- Random strain ---
# Small random lattice strain, up to 1% in each direction.
# This is representative of thermal expansion / elastic fluctuations.
MAX_STRAIN = 0.01  # 1% maximum strain per axis

# --- Read base structure ---
base = ase.io.read('structures/primitive.cif')
print(f"Base structure: {len(base)} atoms, symbols: {base.get_chemical_symbols()}")
print(f"Cell: {base.cell[:].tolist()}")

# INVARIANT 1: nat and ntyp computed from the actual Atoms object
nat_base = len(base)
types_base = set(base.get_chemical_symbols())
ntyp_base = len(types_base)
print(f"Base nat={nat_base}, ntyp={ntyp_base}, types={types_base}")

rng = np.random.default_rng(42)  # fixed seed for reproducibility

for i in range(51):
    nnn = f"{i:03d}"
    atoms = base.copy()

    if i == 0:
        # Unperturbed base structure
        pass
    else:
        # --- Thermal displacements (Einstein model) ---
        # Each atom gets a random displacement drawn from a Gaussian with
        # per-axis RMS = RMS_C (0.05 A at T = 1000 K).
        # U = 0.0025 A^2 is the mean-square displacement per axis.
        # Per-axis RMS = sqrt(U) = 0.05 A.
        displacements = rng.normal(loc=0.0, scale=RMS_C, size=(len(atoms), 3))
        atoms.positions += displacements

        # --- Small random lattice strain ---
        # Apply a random strain tensor with components up to MAX_STRAIN.
        # strain = [[e11, e12, e13], [e21, e22, e23], [e31, e32, e33]]
        # We use a symmetric strain tensor (e12=e21, etc.) for physical realism.
        strain = rng.uniform(-MAX_STRAIN, MAX_STRAIN, size=(3, 3))
        strain = (strain + strain.T) / 2.0  # symmetrize
        # Apply strain to the cell: new_cell = (I + strain) @ old_cell
        # In ASE, cell is a 3x3 matrix of row vectors (a1, a2, a3).
        # The strain acts on the cell vectors:
        #   new_cell = (I + strain) @ cell  (if strain is in the same basis)
        # More precisely, if we define strain in the Cartesian basis:
        #   new_cell = cell @ (I + strain).T
        # We use the standard convention: cell vectors transform as
        #   a_i' = a_i + sum_j strain_ij * a_j
        # which in matrix form (cell as row vectors) is:
        #   new_cell = (I + strain) @ cell
        new_cell = (np.eye(3) + strain) @ atoms.cell[:]
        atoms.set_cell(new_cell, scale_atoms=True)

    # INVARIANT 1: assert nat/ntyp for each structure
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == nat_base, f"Structure {nnn}: nat changed from {nat_base} to {nat}"
    assert ntyp == ntyp_base, f"Structure {nnn}: ntyp changed from {ntyp_base} to {ntyp}"

    out_path = f"structures/structure_{nnn}.cif"
    ase.io.write(out_path, atoms)
    if i == 0 or i % 10 == 0:
        print(f"Wrote {out_path} (nat={nat}, ntyp={ntyp})")

print("Done: wrote 51 structures (structure_000.cif through structure_050.cif)")
