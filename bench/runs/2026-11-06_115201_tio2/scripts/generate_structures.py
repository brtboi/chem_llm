import numpy as np
import ase.io
from ase import Atoms

# =============================================================================
# generate_structures.py
# Writes 51 structures: structure_000.cif (unperturbed) through structure_050.cif
# Perturbations: random atomic displacements (thermal motion) + small random strain
# =============================================================================

# Temperature choice:
# Rutile TiO2 is the thermodynamically stable phase at ambient conditions (T < ~860 K).
# At T = 300 K (room temperature) rutile is fully stable; anatase and brookite are
# metastable. We use T = 300 K as the reference temperature for thermal displacements.
T = 300  # Kelvin

# Species-specific RMS displacement amplitudes (in Angstrom):
# Derived from Debye-Waller factors in rutile TiO2 at 300 K.
# O is lighter (16 amu) than Ti (48 amu) and more weakly bound, so it moves more.
# Typical experimental U_iso values at 300 K:
#   O:  U_iso ~ 0.010-0.015 A^2  ->  RMS = sqrt(U_iso) ~ 0.10-0.12 A
#   Ti: U_iso ~ 0.004-0.006 A^2  ->  RMS = sqrt(U_iso) ~ 0.06-0.08 A
# We use the upper end of these ranges to be conservative:
#   O:  RMS = 0.12 A  (per-axis, i.e. sqrt(U) where U is the mean-square displacement in A^2)
#   Ti: RMS = 0.06 A  (per-axis)
# Note: these are per-axis RMS values. The total displacement magnitude is sqrt(3)*RMS.
RMS_O  = 0.12  # Angstrom, per-axis RMS displacement for O at 300 K
RMS_Ti = 0.06  # Angstrom, per-axis RMS displacement for Ti at 300 K

# Random strain: small isotropic + anisotropic strain, ~1% per axis.
# This represents thermal expansion / lattice distortion.
# Strain tensor components are dimensionless (fractional change in length).
# We apply a random strain tensor with components drawn from N(0, 0.01).
# The resulting cell vectors are: a' = (I + eps) @ a
# where eps is the strain tensor.
STRAIN_SIGMA = 0.01  # 1% standard deviation for each strain component

# Number of perturbed structures (001 through 050)
N_PERTURBED = 50

# Seed for reproducibility
np.random.seed(42)

# Read the base (unperturbed) structure
base = ase.io.read('structures/structure_000.cif')
print(f"Base structure: {len(base)} atoms, cell = {base.cell[:].tolist()}")

# Write structure_000.cif (unperturbed base)
ase.io.write('structures/structure_000.cif', base)
print("Wrote structures/structure_000.cif (unperturbed)")

# Generate perturbed structures 001 through 050
for i in range(1, N_PERTURBED + 1):
    atoms = base.copy()
    
    # --- Apply random strain to the cell ---
    # Strain tensor: symmetric 3x3 matrix with components ~ N(0, STRAIN_SIGMA)
    # This represents a small random lattice distortion.
    # The strain is applied as: new_cell = (I + eps) @ old_cell
    # where eps is the infinitesimal strain tensor.
    eps = np.random.normal(0, STRAIN_SIGMA, (3, 3))
    # Symmetrize to ensure a proper strain tensor (no rotation component)
    eps = 0.5 * (eps + eps.T)
    new_cell = (np.eye(3) + eps) @ atoms.cell[:]
    atoms.set_cell(new_cell, scale_atoms=True)  # scale_atoms=True keeps atoms in fractional coords
    
    # --- Apply random atomic displacements ---
    # Each atom gets a random displacement drawn from N(0, RMS_species^2) per axis.
    # The displacement is in Cartesian coordinates (Angstrom).
    # We use the species-specific RMS values defined above.
    symbols = atoms.get_chemical_symbols()
    displacements = np.zeros((len(atoms), 3))
    for j, sym in enumerate(symbols):
        rms = RMS_O if sym == 'O' else RMS_Ti
        displacements[j] = np.random.normal(0, rms, 3)
    atoms.positions += displacements
    
    # --- INVARIANT 1: assert nat/ntyp consistency ---
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 6, f"Structure {i:03d}: expected 6 atoms, got {nat}"
    assert ntyp == 2, f"Structure {i:03d}: expected 2 species, got {ntyp}"
    
    # Write the structure
    fname = f'structures/structure_{i:03d}.cif'
    ase.io.write(fname, atoms)
    print(f"Wrote {fname}: nat={nat}, ntyp={ntyp}")

print("\nDone. Generated 51 structures (000-050).")
