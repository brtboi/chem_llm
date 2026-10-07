import numpy as np
from ase.io import read, write
from ase import Atoms

# =============================================================================
# generate_structures.py
# Generates 51 structures (000-050) for graphite (mp-48, P6_3/mmc #194).
# Structure 000 is the unperturbed primitive cell.
# Structures 001-050 have random thermal displacements + small random strain.
# =============================================================================

# --- Temperature choice ---
# Graphite is the thermodynamically stable phase of carbon at ambient
# conditions (room temperature up to ~3600 K sublimation at 1 atm).
# We choose T = 300 K as a representative ambient temperature.
# This is well within graphite's stability range.
T = 300.0  # K

# --- Displacement amplitudes ---
# For carbon at 300 K, the mean-square displacement U (in A^2) can be
# estimated from the Einstein model:
#   U = 3*hbar^2 / (2*M*k_B*T) * (Debye integral correction)
# For C (M = 12 amu), Debye temperature ~1500 K, at T=300 K:
#   U ~ 0.002 A^2  (mean-square displacement per axis)
#   RMS per axis = sqrt(U) ~ 0.045 A
# We use RMS = 0.045 A as the per-axis displacement amplitude.
# This is a single-species system (only C), so one amplitude suffices.
# The displacement is drawn from a Gaussian with this RMS per axis.
RMS_DISPLACEMENT = 0.045  # A, per-axis RMS displacement (sqrt(U) where U ~ 0.002 A^2)

# --- Strain ---
# Small random lattice strain representative of thermal lattice fluctuations.
# We apply a random strain tensor with components ~0.5% (0.005) in magnitude.
# This is isotropic + small shear, well within the elastic regime.
STRAIN_MAGNITUDE = 0.005  # dimensionless, ~0.5% strain

# --- Seed for reproducibility ---
np.random.seed(42)

# Read the primitive cell (already reduced in reduce_primitive.py)
base = read('structures/base_C_prim.cif')
print(f'Base primitive cell: nat = {len(base)}, ntyp = {len(set(base.get_chemical_symbols()))}')

# INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
nat = len(base)
types = set(base.get_chemical_symbols())
ntyp = len(types)
print(f'Computed: nat = {nat}, ntyp = {ntyp}, types = {types}')

# Structure 000: unperturbed base
write('structures/structure_000.cif', base)
print('Wrote structures/structure_000.cif (unperturbed)')

# Structures 001-050: perturbed copies
for i in range(1, 51):
    atoms = base.copy()
    
    # 1. Random atomic displacements (thermal motion)
    # Displacements in Cartesian coordinates (A), Gaussian with RMS per axis
    displacements = np.random.normal(0.0, RMS_DISPLACEMENT, size=(len(atoms), 3))
    atoms.positions += displacements
    
    # 2. Small random lattice strain
    # Generate a random strain tensor with components ~STRAIN_MAGNITUDE
    strain = np.random.normal(0.0, STRAIN_MAGNITUDE, size=(3, 3))
    # Make it symmetric (strain tensor is symmetric)
    strain = 0.5 * (strain + strain.T)
    # Apply strain to the cell: new_cell = (I + strain) @ old_cell
    # In ASE, cell is stored as rows (a1, a2, a3), so we apply:
    # new_cell_array = (I + strain) @ cell_array  (strain acts on the basis vectors)
    new_cell = (np.eye(3) + strain) @ atoms.cell.array
    atoms.cell = new_cell
    
    # INVARIANT 1: assert nat/ntyp for each structure
    assert len(atoms) == nat, f'structure {i:03d}: nat changed from {nat} to {len(atoms)}'
    assert len(set(atoms.get_chemical_symbols())) == ntyp, f'structure {i:03d}: ntyp changed'
    
    fname = f'structures/structure_{i:03d}.cif'
    write(fname, atoms)
    
    if i <= 3 or i == 50:
        print(f'Wrote {fname}: nat={len(atoms)}, ntyp={len(set(atoms.get_chemical_symbols()))}')

print('Done: wrote 51 structures (000-050)')
