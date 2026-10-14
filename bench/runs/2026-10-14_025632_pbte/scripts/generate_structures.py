import numpy as np
import ase.io
from ase import Atoms

# =============================================================================
# generate_structures.py
# Generates 51 PbTe structures: structure_000.cif (unperturbed) and
# structure_001..050.cif (perturbed with thermal displacements + random strain)
#
# Temperature choice: 300 K
#   PbTe is a stable rock-salt semiconductor at room temperature.
#   Materials Project reports a band gap of ~0.806 eV for mp-19717,
#   confirming it is a stable phase at ambient conditions.
#   300 K is therefore a representative temperature for this phase.
#
# Thermal displacement amplitudes (per-axis RMS, in Angstrom):
#   The per-axis RMS displacement is sigma = sqrt(U), where U is the
#   crystallographic mean-square displacement parameter in A^2.
#   We use the Einstein model: U = 3*k_B*T / (M * omega^2)
#   For a rough estimate, we use typical Debye temperatures:
#     Pb:  theta_D ~ 105 K  ->  omega ~ k_B*theta_D/hbar
#     Te:  theta_D ~ 165 K
#   A simpler approach: use known U values from literature at 300 K.
#   For PbTe at 300 K, typical U values are approximately:
#     Pb: U ~ 0.008 A^2  ->  sigma_Pb = sqrt(0.008) ~ 0.089 A
#     Te: U ~ 0.005 A^2  ->  sigma_Te = sqrt(0.005) ~ 0.071 A
#   Pb is heavier and more weakly bound, so it moves more (larger sigma).
#   We use sigma_Pb = 0.09 A, sigma_Te = 0.07 A as per-axis RMS displacements.
#
# Random strain: small isotropic + shear strain, magnitude ~0.5%
#   This represents lattice distortion from thermal expansion and
#   local strain fields.
# =============================================================================

np.random.seed(42)  # reproducibility

# Read the primitive cell
atoms = ase.io.read('structures/prim_PbTe.cif')
print(f"Base structure: {len(atoms)} atoms, symbols: {atoms.get_chemical_symbols()}")
print(f"Cell: {atoms.cell.array}")

# Per-species RMS displacement amplitudes (in Angstrom)
# sigma = sqrt(U), where U is the mean-square displacement in A^2
# Pb: heavier, more weakly bound -> larger displacement
# Te: lighter, more strongly bound -> smaller displacement
SIGMA = {
    'Pb': 0.09,  # A, per-axis RMS displacement (sqrt of U_Pb ~ 0.0081 A^2)
    'Te': 0.07,  # A, per-axis RMS displacement (sqrt of U_Te ~ 0.0049 A^2)
}

# Strain magnitude: 0.5% (typical for thermal strain in a solid)
STRAIN_MAG = 0.005

# Write structure_000: unperturbed base
ase.io.write('structures/structure_000.cif', atoms)
print("Wrote structures/structure_000.cif (unperturbed)")

# Generate 50 perturbed structures
for i in range(1, 51):
    perturbed = atoms.copy()
    
    # Apply random atomic displacements (thermal motion)
    # Each atom gets a Gaussian displacement with per-species sigma
    positions = perturbed.get_positions().copy()
    for idx, (pos, symbol) in enumerate(zip(positions, perturbed.get_chemical_symbols())):
        sigma = SIGMA[symbol]
        # Gaussian displacement in Cartesian coordinates (Angstrom)
        disp = np.random.normal(0.0, sigma, size=3)
        positions[idx] += disp
    perturbed.set_positions(positions)
    
    # Apply small random lattice strain
    # Random strain tensor: isotropic + shear components, magnitude ~STRAIN_MAG
    # Strain tensor is symmetric, trace can be nonzero (volumetric change)
    strain = np.random.normal(0.0, STRAIN_MAG, size=(3, 3))
    strain = 0.5 * (strain + strain.T)  # make symmetric
    
    # Apply strain to the cell
    # new_cell = (I + strain) @ old_cell
    old_cell = perturbed.cell.array.copy()
    new_cell = (np.eye(3) + strain) @ old_cell
    perturbed.cell = new_cell
    
    # INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
    nat = len(perturbed)
    ntyp = len(set(perturbed.get_chemical_symbols()))
    assert nat == 2, f"Expected 2 atoms, got {nat}"
    assert ntyp == 2, f"Expected 2 species, got {ntyp}"
    
    # Write the perturbed structure
    fname = f'structures/structure_{i:03d}.cif'
    ase.io.write(fname, perturbed)
    
    if i <= 3 or i == 50:
        print(f"Wrote {fname}: nat={nat}, ntyp={ntyp}, "
              f"strain_diag={np.diag(strain).round(5)}")

print("\nDone: wrote 51 structures (structure_000.cif through structure_050.cif)")
