import numpy as np
import ase.io
from ase import Atoms
from ase.build import fcc111

# =============================================================================
# generate_structures.py
# Generates 51 perturbed fcc-Al structures (structure_000.cif ... structure_050.cif)
# =============================================================================

# --- Temperature choice ---
# fcc Al is the stable phase at room temperature (298 K) and up to the melting
# point (~933 K). We choose T = 300 K, well within the stable fcc range.
# At 300 K the Debye-Waller (mean-square) displacement of Al is small.
# The Debye temperature of Al is ~428 K, so at T=300 K the thermal motion is
# in the classical-ish regime but still modest.
# We use a simple Einstein-Debye estimate for the RMS displacement:
#   <u^2> = (3 * hbar^2) / (2 * pi^2 * m * k_B * T_D) * (T/T_D)^2  (low-T limit)
# But for a practical DFT perturbation we use a more direct approach:
#   The mean-square displacement U (in A^2) for Al at 300 K is approximately
#   U ~ 0.001 A^2 (from experimental Debye-Waller factors).
#   RMS displacement per axis = sqrt(U) ~ 0.032 A.
#
# We use a slightly larger amplitude (0.05 A RMS per axis) to ensure the
# perturbation is meaningful for the DFT calculation while still being
# physically reasonable for thermal motion at ~300 K.

T_K = 300  # Kelvin

# Per-species RMS displacement amplitudes (in Angstrom, per axis)
# Al is the only species. Lighter atoms move more; Al is light (Z=13).
# We use 0.05 A RMS per axis as a representative thermal displacement at 300 K.
# This is the RMS (root-mean-square) displacement, i.e. sqrt(U) where U is
# the crystallographic mean-square displacement parameter in A^2.
# So U = (0.05 A)^2 = 0.0025 A^2.
RMS_DISP_AL = 0.05  # Angstrom, RMS per axis = sqrt(U)

# Small random lattice strain: up to 1% strain in each direction
MAX_STRAIN = 0.01  # dimensionless (1%)

# Seed for reproducibility
np.random.seed(42)

# --- Read the base (primitive) structure ---
base = ase.io.read('structures/Al_primitive.cif')
print(f"Base structure: {len(base)} atoms, cell = {base.cell[:].tolist()}")

# INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
nat_base = len(base)
types_base = set(base.get_chemical_symbols())
ntyp_base = len(types_base)
print(f"Base: nat={nat_base}, ntyp={ntyp_base}, species={types_base}")
assert nat_base == 1, f"Expected 1 atom in primitive Al cell, got {nat_base}"
assert ntyp_base == 1, f"Expected 1 species, got {ntyp_base}"

# --- Generate 51 structures ---
for i in range(51):
    if i == 0:
        # structure_000: unperturbed base
        atoms = base.copy()
    else:
        atoms = base.copy()
        
        # --- Random atomic displacements (thermal motion) ---
        # Displace each atom by a random vector with Gaussian distribution,
        # RMS = RMS_DISP_AL per axis.
        # Displacements are in Cartesian coordinates (Angstrom).
        # We use fractional coordinates to keep the atom inside the cell.
        disp = np.random.normal(0.0, RMS_DISP_AL, size=(nat_base, 3))
        # Convert Cartesian displacement to fractional displacement
        # fractional_disp = disp @ inv(cell)
        inv_cell = np.linalg.inv(atoms.cell[:])
        frac_disp = disp @ inv_cell
        # Apply displacement in fractional coordinates
        scaled_pos = atoms.get_scaled_positions()
        scaled_pos += frac_disp
        # Wrap into [0, 1) to keep atoms in the unit cell
        scaled_pos -= np.floor(scaled_pos)
        atoms.set_scaled_positions(scaled_pos)
        
        # --- Small random lattice strain ---
        # Apply a random strain tensor with components in [-MAX_STRAIN, MAX_STRAIN]
        strain = np.random.uniform(-MAX_STRAIN, MAX_STRAIN, size=(3, 3))
        # Make strain symmetric (physical strain is symmetric)
        strain = 0.5 * (strain + strain.T)
        # New cell = (I + strain) @ old_cell
        new_cell = (np.eye(3) + strain) @ atoms.cell[:]
        atoms.set_cell(new_cell, scale_atoms=True)
    
    # INVARIANT 1: assert nat/ntyp for each structure
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == nat_base, f"Structure {i:03d}: nat changed from {nat_base} to {nat}"
    assert ntyp == ntyp_base, f"Structure {i:03d}: ntyp changed from {ntyp_base} to {ntyp}"
    
    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)
    print(f"Wrote {out_path}: nat={nat}, ntyp={ntyp}")

print("\nDone. Generated 51 structures.")
