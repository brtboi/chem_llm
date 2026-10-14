import numpy as np
import ase.io

# =============================================================================
# generate_structures.py
# Generates 51 perturbed hBN structures (structure_000.cif ... structure_050.cif)
# structure_000 is the unperturbed base; 001-050 have random thermal displacements
# and small random lattice strain.
# =============================================================================

# --- Temperature choice -------------------------------------------------------
# hBN (hexagonal boron nitride, P6_3/mmc) is the ground-state phase of BN at
# ambient pressure and is stable from 0 K up to its sublimation temperature
# (~3000 K in inert atmosphere). 300 K (room temperature) is well within the
# stable range and is the natural choice for a thermal-displacement study.
T = 300.0  # Kelvin

# --- Displacement amplitudes --------------------------------------------------
# We use empirical thermal displacement parameters from X-ray/neutron diffraction
# of hBN at 300 K:
#   B: U_iso ~ 0.005 A^2  -> RMS = sqrt(0.005) ~ 0.071 A
#   N: U_iso ~ 0.004 A^2  -> RMS = sqrt(0.004) ~ 0.063 A
# These are typical values for light atoms in a stiff lattice at room temperature.
# B is lighter than N, so it has a slightly larger displacement.
#
# U is the crystallographic displacement parameter (mean-square, A^2).
# The per-axis RMS is sqrt(U) in A.

RMS_B = 0.071  # A, per-axis RMS displacement for B (U_B = 0.005 A^2)
RMS_N = 0.063  # A, per-axis RMS displacement for N (U_N = 0.004 A^2)

# --- Lattice strain -----------------------------------------------------------
# Small random lattice strain to represent thermal expansion / anharmonic effects.
# At 300 K, the thermal expansion of hBN is ~1e-5 K^-1, so over a 300 K range
# the strain is ~3e-3. We use a maximum strain of 0.005 (0.5%) to be safe.
MAX_STRAIN = 0.005  # dimensionless, maximum strain per axis

# --- Read base structure ------------------------------------------------------
base = ase.io.read('structures/structure_000.cif')
print(f"Base structure: {len(base)} atoms, symbols: {base.get_chemical_symbols()}")

# --- Generate perturbed structures --------------------------------------------
np.random.seed(42)  # reproducibility

for i in range(1, 51):
    atoms = base.copy()
    
    # Apply random lattice strain
    # strain tensor is symmetric, small, random
    strain = np.random.uniform(-MAX_STRAIN, MAX_STRAIN, size=(3, 3))
    strain = (strain + strain.T) / 2.0  # symmetrize
    # Apply strain to the cell: new_cell = (I + strain) @ old_cell
    old_cell = np.array(atoms.cell)
    new_cell = (np.eye(3) + strain) @ old_cell
    atoms.set_cell(new_cell, scale_atoms=True)  # scale atoms to follow the cell
    
    # Apply random atomic displacements (thermal motion)
    # Displacements are in Cartesian coordinates (Angstrom)
    # Each atom gets a random displacement drawn from a Gaussian with
    # standard deviation = RMS for that species.
    symbols = atoms.get_chemical_symbols()
    positions = np.array(atoms.positions)
    for j, sym in enumerate(symbols):
        if sym == 'B':
            rms = RMS_B
        elif sym == 'N':
            rms = RMS_N
        else:
            raise ValueError(f"Unknown species: {sym}")
        # Gaussian displacement in each Cartesian direction
        disp = np.random.normal(0.0, rms, size=3)
        positions[j] += disp
    atoms.positions = positions
    
    # INVARIANT 1: nat and ntyp must match the actual structure
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 4, f"Structure {i:03d}: expected nat=4, got {nat}"
    assert ntyp == 2, f"Structure {i:03d}: expected ntyp=2, got {ntyp}"
    
    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)
    print(f"Wrote {out_path}: nat={nat}, ntyp={ntyp}")

print("\nDone. Generated 51 structures (000-050).")
