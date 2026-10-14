import numpy as np
import ase.io
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

# =============================================================================
# generate_structures.py
# Generate 51 wurtzite GaN structures: structure_000.cif (unperturbed base)
# and structure_001.cif ... structure_050.cif (perturbed copies).
#
# Perturbations:
#   1. Random atomic displacements representative of thermal motion.
#   2. Small random lattice strain.
#
# Temperature choice:
#   Wurtzite GaN is the stable phase at room temperature and up to very high
#   temperatures (it is the ground-state phase of GaN at ambient pressure).
#   We choose T = 300 K (room temperature) because:
#     - Wurtzite GaN is the experimentally observed stable phase at 300 K.
#     - Thermal displacements at 300 K are small enough to preserve the
#       wurtzite symmetry on average while being physically meaningful.
#     - The Debye-Waller factors (U values) at 300 K are well-documented
#       for GaN: Ga has U ~ 0.005-0.01 A^2, N has U ~ 0.003-0.008 A^2.
#
# Displacement amplitudes (RMS, in Angstrom):
#   The crystallographic displacement parameter U is the mean-square
#   displacement in A^2. The per-axis RMS displacement is sqrt(U) in A.
#   At T = 300 K, typical values for wurtzite GaN are:
#     Ga: U ~ 0.008 A^2  -> RMS = sqrt(0.008) ~ 0.089 A
#     N:  U ~ 0.005 A^2  -> RMS = sqrt(0.005) ~ 0.071 A
#   N is lighter (14 amu vs 70 amu) and more weakly bound in the sense that
#   it has a higher vibrational frequency, but in wurtzite GaN the Ga atoms
#   actually have slightly larger displacements due to their lower force
#   constants. We use:
#     Ga: RMS = 0.089 A  (U = 0.008 A^2)
#     N:  RMS = 0.071 A  (U = 0.005 A^2)
#   These are per-axis RMS values. For 3D Gaussian displacements, each
#   Cartesian component is drawn from N(0, RMS^2).
#
# Lattice strain:
#   Small random strain tensor with components drawn from N(0, 0.005^2),
#   i.e. RMS strain of 0.5%. This is well within the elastic stability
#   range of wurtzite GaN.
# =============================================================================

np.random.seed(42)  # Reproducibility

# Read the base (primitive) structure
base = ase.io.read('structures/gan_primitive.cif')
print(f"Base structure: {len(base)} atoms, cell =")
print(base.cell)

# Displacement parameters (RMS per axis, in Angstrom)
# Derived from U (mean-square displacement in A^2) at T = 300 K:
#   Ga: U = 0.008 A^2 -> RMS = sqrt(0.008) = 0.0894 A
#   N:  U = 0.005 A^2 -> RMS = sqrt(0.005) = 0.0707 A
rms_displacements = {
    'Ga': np.sqrt(0.008),  # 0.0894 A, per-axis RMS
    'N':  np.sqrt(0.005),  # 0.0707 A, per-axis RMS
}

# Strain RMS (dimensionless, i.e. fractional change in lattice parameter)
strain_rms = 0.005  # 0.5% RMS strain

N_STRUCTURES = 51  # 000 (unperturbed) + 001-050 (perturbed)

for i in range(N_STRUCTURES):
    atoms = base.copy()
    
    if i == 0:
        # Unperturbed base structure
        pass
    else:
        # Apply random lattice strain
        # Strain tensor: symmetric 3x3 matrix with small random components
        strain = np.random.normal(0.0, strain_rms, size=(3, 3))
        # Make it symmetric (physical strain tensor is symmetric)
        strain = 0.5 * (strain + strain.T)
        # New cell = (I + strain) @ old_cell
        # Note: cell vectors are row vectors in ASE, so we apply strain as:
        # new_cell = (I + strain) @ old_cell  (strain acts on the basis vectors)
        new_cell = (np.eye(3) + strain) @ atoms.cell
        atoms.cell = new_cell
        
        # Apply random atomic displacements (thermal motion)
        # Each atom gets a 3D Gaussian displacement with species-specific RMS
        for idx, symbol in enumerate(atoms.get_chemical_symbols()):
            rms = rms_displacements[symbol]
            # Displacement in Cartesian coordinates (Angstrom)
            disp = np.random.normal(0.0, rms, size=3)
            atoms.positions[idx] += disp
    
    # INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 4, f"Structure {i:03d}: expected nat=4, got {nat}"
    assert ntyp == 2, f"Structure {i:03d}: expected ntyp=2, got {ntyp}"
    
    # Write the structure
    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)
    
    if i < 3 or i == N_STRUCTURES - 1:
        print(f"Wrote {out_path}: nat={nat}, ntyp={ntyp}")

print(f"\nGenerated {N_STRUCTURES} structures: structure_000.cif to structure_{N_STRUCTURES-1:03d}.cif")
print("Done.")
