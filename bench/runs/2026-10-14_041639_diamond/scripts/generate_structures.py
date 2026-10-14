import numpy as np
import ase.io
import os

# =============================================================================
# generate_structures.py
# Generates 51 diamond-C structures: structure_000.cif (unperturbed) and
# structure_001.cif .. structure_050.cif (perturbed with thermal displacements
# and small random lattice strain).
#
# Temperature choice: 300 K (room temperature).
#   Diamond is the thermodynamically stable phase of carbon at ambient
#   pressure and room temperature. Graphite is the ground state at 0 K,
#   but diamond is kinetically stable and the standard reference phase
#   for DFT band-structure calculations at room temperature.
#
# Displacement amplitudes (crystallographic U parameters):
#   For diamond C at 300 K, the Debye-Waller factor gives U ~ 0.005 A^2.
#   The per-axis RMS displacement is sqrt(U) ~ 0.07 A.
#   We use sigma = 0.07 A as the standard deviation of the Gaussian
#   displacement in each Cartesian direction.
#
# Lattice strain:
#   Small random strain of order 0.5% (0.005) to represent lattice
#   distortion from thermal expansion / anharmonic effects.
# =============================================================================

np.random.seed(42)

# Read the primitive cell (already reduced in reduce_cell.py)
base = ase.io.read('structures/c_base_prim.cif')
print(f"Base structure: nat={len(base)}, ntyp={len(set(base.get_chemical_symbols()))}")
print(f"Cell vectors (A):\n{base.cell[:]}\n")

# Displacement parameters
# U = 0.005 A^2 (mean-square displacement, crystallographic convention)
# sigma = sqrt(U) = 0.07 A (per-axis RMS displacement)
U_C = 0.005  # A^2
sigma_C = np.sqrt(U_C)  # A ~ 0.07 A

# Strain magnitude (small random strain, ~0.5%)
strain_mag = 0.005

# Create output directory
os.makedirs('structures', exist_ok=True)

# Structure 000: unperturbed base
ase.io.write('structures/structure_000.cif', base)
print("Wrote structures/structure_000.cif (unperturbed)")

# Structures 001-050: perturbed
for i in range(1, 51):
    atoms = base.copy()
    
    # Apply random lattice strain
    # Generate a small random symmetric strain tensor
    strain = np.random.randn(3, 3) * strain_mag
    strain = (strain + strain.T) / 2.0  # make symmetric
    
    # Apply strain to the cell
    # new_cell = (I + strain) @ old_cell
    new_cell = (np.eye(3) + strain) @ atoms.cell[:]
    atoms.set_cell(new_cell, scale_atoms=True)
    
    # Apply random atomic displacements (thermal motion)
    # Displacements in Cartesian coordinates, Gaussian with std = sigma_C
    displacements = np.random.randn(len(atoms), 3) * sigma_C
    atoms.positions += displacements
    
    # INVARIANT 1: nat and ntyp must match the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Structure {i:03d}: expected nat=2, got {nat}"
    assert ntyp == 1, f"Structure {i:03d}: expected ntyp=1, got {ntyp}"
    
    fname = f'structures/structure_{i:03d}.cif'
    ase.io.write(fname, atoms)
    
    if i <= 3 or i == 50:
        print(f"Wrote {fname}: nat={nat}, ntyp={ntyp}, "
              f"strain={strain_mag:.4f}, sigma={sigma_C:.4f} A")

print("\nDone. Generated 51 structures (000-050).")
print(f"Displacement parameters: U_C = {U_C} A^2, sigma_C = {sigma_C:.4f} A")
print(f"Strain magnitude: {strain_mag} ({strain_mag*100:.1f}%)")
