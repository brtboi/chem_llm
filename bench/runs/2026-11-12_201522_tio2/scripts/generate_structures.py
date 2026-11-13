"""
Generate 51 perturbed TiO2 (rutile) structures for DFT band-structure calculations.

Structure 000: unperturbed primitive cell (6 atoms, P4_2/mnm #136).
Structures 001-050: random thermal displacements + small random lattice strain.

Temperature choice:
  Rutile TiO2 is the stable phase at room temperature (298 K) up to ~1000 K.
  We use T = 300 K, well within the rutile stability range.

Thermal displacement model:
  For each atom, the mean-square displacement (crystallographic U) is:
      U = (3 * k_B * T) / (2 * pi^2 * m * omega^2)
  In practice we use a Debye-like estimate.  For TiO2 at 300 K:
    - Ti (mass 47.87 amu): U_Ti ~ 0.003 A^2  -> RMS = sqrt(U) ~ 0.055 A
    - O  (mass 15.99 amu): U_O  ~ 0.008 A^2  -> RMS = sqrt(U) ~ 0.090 A
  (Lighter O atoms vibrate more; these values are consistent with
   experimental Debye-Waller factors for rutile TiO2 at 300 K.)

  We draw displacements from a 3D Gaussian with per-axis std = sqrt(U/3)
  (isotropic approximation), so the total mean-square displacement is U.

Lattice strain:
  Small random strain tensor with components drawn from N(0, 0.005)
  (0.5% strain), applied to the cell vectors.

Units:
  - U values are in A^2 (mean-square displacement).
  - RMS displacement per axis = sqrt(U/3) in A.
  - Strain is dimensionless.
"""

import numpy as np
import ase.io
from ase import Atoms
from ase.spacegroup import crystal

# --- Parameters ---
T = 300.0  # K

# Mean-square displacements U in A^2 (crystallographic, total over 3 axes)
# Derived from Debye-Waller factors for rutile TiO2 at 300 K:
#   Ti: U ~ 0.003 A^2  (heavier, stiffer bond)
#   O:  U ~ 0.008 A^2  (lighter, more mobile)
U_Ti = 0.003  # A^2
U_O = 0.008   # A^2

# Per-axis RMS displacement = sqrt(U/3) in A
rms_Ti = np.sqrt(U_Ti / 3.0)  # ~0.032 A per axis
rms_O = np.sqrt(U_O / 3.0)    # ~0.052 A per axis

# Lattice strain std (dimensionless)
strain_std = 0.005  # 0.5%

N_STRUCTURES = 51  # 000 through 050

# --- Read base structure (primitive cell, 6 atoms) ---
base = ase.io.read('structures/structure_000.cif')
print(f"Base structure: {len(base)} atoms, cell = {base.get_cell()}")

# INVARIANT 1: nat and ntyp from the actual Atoms object
nat = len(base)
typ_set = set(base.get_chemical_symbols())
ntyp = len(typ_set)
print(f"nat = {nat}, ntyp = {ntyp}, species = {sorted(typ_set)}")
assert nat == 6, f"Expected 6 atoms in primitive rutile cell, got {nat}"
assert ntyp == 2, f"Expected 2 species (Ti, O), got {ntyp}"

# Map species -> RMS displacement (per axis, in A)
rms_map = {'Ti': rms_Ti, 'O': rms_O}

rng = np.random.default_rng(seed=42)

for i in range(N_STRUCTURES):
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()
        
        # --- Random lattice strain ---
        # Apply a small random strain to the cell vectors.
        # strain tensor is symmetric, traceless-ish (small diagonal + off-diagonal)
        strain = rng.normal(0.0, strain_std, size=(3, 3))
        # Make it symmetric
        strain = 0.5 * (strain + strain.T)
        # Apply to cell: new_cell = (I + strain) @ old_cell
        old_cell = np.array(atoms.get_cell())
        new_cell = (np.eye(3) + strain) @ old_cell
        atoms.set_cell(new_cell, scale_atoms=True)
        
        # --- Random atomic displacements (thermal motion) ---
        # Each atom gets a 3D Gaussian displacement with per-axis std = rms[species]
        # The total mean-square displacement per atom is U = 3 * rms^2 (in A^2)
        symbols = atoms.get_chemical_symbols()
        displacements = np.zeros((len(atoms), 3))
        for j, sym in enumerate(symbols):
            std = rms_map[sym]  # per-axis RMS in A
            displacements[j] = rng.normal(0.0, std, size=3)
        
        # Add displacements in Cartesian coordinates
        atoms.positions += displacements
    
    # INVARIANT 1: verify nat/ntyp for each structure
    assert len(atoms) == nat, f"Structure {i:03d}: nat changed from {nat}"
    assert len(set(atoms.get_chemical_symbols())) == ntyp, \
        f"Structure {i:03d}: ntyp changed from {ntyp}"
    
    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)

print(f"\nWrote {N_STRUCTURES} structures: structure_000.cif through structure_{N_STRUCTURES-1:03d}.cif")
print(f"Displacement parameters (T={T} K):")
print(f"  Ti: U = {U_Ti} A^2, per-axis RMS = {rms_Ti:.4f} A")
print(f"  O:  U = {U_O} A^2, per-axis RMS = {rms_O:.4f} A")
print(f"  Lattice strain std: {strain_std} (dimensionless)")
