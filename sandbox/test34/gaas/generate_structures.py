import numpy as np
import ase.io
from ase import Atoms
import os

# ============================================================
# generate_structures.py
# Generates 51 perturbed GaAs structures (structure_000 to structure_050)
# structure_000 = unperturbed base
# structures 001-050 = random thermal displacements + small random lattice strain
# ============================================================

# Temperature choice:
# GaAs is stable at room temperature and up to its melting point (~1511 K).
# We choose T = 300 K (room temperature) as the reference temperature for
# thermal displacements. This is well within the stable range of the
# zincblende phase of GaAs.
T = 300.0  # K

# Debye-Waller factors (crystallographic displacement parameters U in A^2):
# U is the mean-square displacement; per-axis RMS = sqrt(U) in A.
# Lighter atoms (Ga, Z=31) move more than heavier atoms (As, Z=33).
# Typical U values for GaAs at 300 K from literature:
#   Ga: U ~ 0.005 A^2  -> RMS ~ 0.071 A
#   As: U ~ 0.003 A^2  -> RMS ~ 0.055 A
# We use these as the per-axis RMS displacements (sigma) for Gaussian sampling.
U_Ga = 0.005   # A^2, mean-square displacement for Ga
U_As = 0.003   # A^2, mean-square displacement for As
sigma_Ga = np.sqrt(U_Ga)  # A, per-axis RMS displacement for Ga
sigma_As = np.sqrt(U_As)  # A, per-axis RMS displacement for As

# Lattice strain: small random strain tensor, magnitude ~0.1%
# This represents thermal expansion/contraction fluctuations.
strain_magnitude = 0.001  # 0.1% strain

# Random seed for reproducibility
np.random.seed(42)

# Read base structure
base = ase.io.read('structures/GaAs_primitive.cif')
print(f"Base structure: {len(base)} atoms, symbols: {base.get_chemical_symbols()}")
print(f"Base cell: {base.get_cell()}")

# Verify nat/ntyp invariant for base
nat_base = len(base)
ntyp_base = len(set(base.get_chemical_symbols()))
print(f"Base nat={nat_base}, ntyp={ntyp_base}")
assert nat_base == 2, f"Expected 2 atoms, got {nat_base}"
assert ntyp_base == 2, f"Expected 2 types, got {ntyp_base}"

os.makedirs('structures', exist_ok=True)

# Write structure_000 (unperturbed)
ase.io.write('structures/structure_000.cif', base)
print("Wrote structures/structure_000.cif (unperturbed)")

for n in range(1, 51):
    atoms = base.copy()
    
    # Apply random lattice strain
    # Generate a small random symmetric strain tensor
    strain = np.random.randn(3, 3) * strain_magnitude
    strain = (strain + strain.T) / 2.0  # make symmetric
    # Apply strain to the cell: new_cell = (I + strain) @ old_cell
    old_cell = atoms.get_cell()
    new_cell = (np.eye(3) + strain) @ np.array(old_cell)
    atoms.set_cell(new_cell, scale_atoms=True)
    
    # Apply random atomic displacements (thermal motion)
    # Each atom gets a Gaussian displacement with species-specific sigma
    positions = atoms.get_positions().copy()
    for i, symbol in enumerate(atoms.get_chemical_symbols()):
        if symbol == 'Ga':
            sigma = sigma_Ga
        elif symbol == 'As':
            sigma = sigma_As
        else:
            raise ValueError(f"Unknown species: {symbol}")
        # Displacement in Cartesian coordinates (A)
        disp = np.random.randn(3) * sigma
        positions[i] += disp
    atoms.set_positions(positions)
    
    # INVARIANT 1: verify nat/ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Structure {n:03d}: expected 2 atoms, got {nat}"
    assert ntyp == 2, f"Structure {n:03d}: expected 2 types, got {ntyp}"
    
    fname = f'structures/structure_{n:03d}.cif'
    ase.io.write(fname, atoms)
    print(f"Wrote {fname} (nat={nat}, ntyp={ntyp})")

print("\nDone. Generated 51 structures (000-050).")
print(f"Temperature: {T} K")
print(f"Ga: U={U_Ga} A^2, RMS={sigma_Ga:.4f} A")
print(f"As: U={U_As} A^2, RMS={sigma_As:.4f} A")
print(f"Strain magnitude: {strain_magnitude} (0.1%)")
