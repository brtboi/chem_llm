import numpy as np
from ase.io import read, write
from ase import Atoms

# =============================================================================
# generate_structures.py
# Generates 51 perturbed 2H MoS2 structures (structure_000.cif .. structure_050.cif)
# 000 = unperturbed base; 001-050 = random thermal displacements + small strain
# =============================================================================

# --- Temperature choice ---
# 2H MoS2 is the thermodynamically stable phase of bulk MoS2 at room temperature.
# It is the ground-state phase from 0 K up to ~1100 K, above which it transforms
# to the 1T (octahedral) phase. We choose T = 300 K, well within the stability
# range of the 2H phase.
T = 300.0  # Kelvin

# --- Displacement amplitudes (Einstein model) ---
# For a harmonic oscillator, the mean-square displacement is <u^2> = 3kBT / k_spring
# where k_spring is the effective spring constant. We use literature-typical
# crystallographic displacement parameters (ADPs) for 2H MoS2 at 300 K:
#
#   Mo: U = 0.003 A^2  (heavier atom, mass 95.95 amu, more strongly bound)
#   S:  U = 0.008 A^2  (lighter atom, mass 32.06 amu, more weakly bound)
#
# U is the mean-square displacement in A^2 (crystallographic convention).
# The per-axis RMS displacement is sqrt(U) in A:
#   Mo: sqrt(0.003) = 0.0548 A
#   S:  sqrt(0.008) = 0.0894 A
#
# We draw displacements from a Gaussian with sigma = sqrt(U/3) per axis, so that
# the total mean-square displacement <u^2> = 3 * sigma^2 = U.
U_Mo = 0.003  # A^2, mean-square displacement for Mo
U_S = 0.008   # A^2, mean-square displacement for S
sigma_Mo = np.sqrt(U_Mo / 3.0)  # A, per-axis RMS for Mo
sigma_S = np.sqrt(U_S / 3.0)    # A, per-axis RMS for S

# --- Random strain ---
# Small random lattice strain: ~0.5% per axis, isotropic + small shear.
# This represents thermal expansion/contraction and small anisotropic distortions.
strain_max = 0.005  # 0.5% maximum strain per component

# --- Read the base structure ---
base = read('structures/MoS2_primitive.cif')
n_base = len(base)
ntyp_base = len(set(base.get_chemical_symbols()))
print(f"Base structure: {n_base} atoms, {ntyp_base} species")
print(f"Cell:\n{base.cell}")
print(f"Atoms: {base.get_chemical_symbols()}")

# --- Generate 51 structures ---
np.random.seed(42)  # reproducibility

for i in range(51):
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()
        
        # Apply random thermal displacements in Cartesian coordinates
        # Each atom gets a Gaussian displacement with species-specific sigma
        for atom in atoms:
            if atom.symbol == 'Mo':
                disp = np.random.normal(0.0, sigma_Mo, 3)
            elif atom.symbol == 'S':
                disp = np.random.normal(0.0, sigma_S, 3)
            else:
                raise ValueError(f"Unexpected element: {atom.symbol}")
            atom.position += disp
        
        # Apply small random lattice strain
        # strain tensor: symmetric 3x3, components in [-strain_max, strain_max]
        strain = np.random.uniform(-strain_max, strain_max, (3, 3))
        # Make it symmetric (physical strain tensor is symmetric)
        strain = 0.5 * (strain + strain.T)
        # New cell = (I + strain) @ old_cell
        new_cell = (np.eye(3) + strain) @ base.cell[:]
        atoms.cell = new_cell
        # Keep positions in the new cell (wrap to [0,1) in fractional coords)
        atoms.wrap()
    
    # INVARIANT 1: assert nat/ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == n_base, f"Structure {i:03d}: nat changed from {n_base} to {nat}"
    assert ntyp == ntyp_base, f"Structure {i:03d}: ntyp changed from {ntyp_base} to {ntyp}"
    
    # Write the structure
    out_path = f'structures/structure_{i:03d}.cif'
    write(out_path, atoms)
    
    if i == 0:
        print(f"Wrote {out_path} (unperturbed base)")
    else:
        # Report the strain magnitude for this structure
        strain_mag = np.linalg.norm(strain)
        print(f"Wrote {out_path} (strain magnitude: {strain_mag:.4f})")

print(f"\nDone. Generated 51 structures in structures/")
