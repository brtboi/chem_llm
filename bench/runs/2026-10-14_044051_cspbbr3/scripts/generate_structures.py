"""
Generate 51 perturbed CsPbBr3 structures (structure_000.cif .. structure_050.cif).

Structure 000: unperturbed primitive cell (from structures/base_primitive.cif).
Structures 001-050: random thermal displacements + small random lattice strain.

Temperature choice: T = 400 K
  Cubic CsPbBr3 (Pm-3m) is the stable phase above ~300 K at ambient pressure.
  Below ~300 K it transforms to orthorhombic (Pbnm). 400 K is safely in the
  cubic stability range and gives thermally meaningful displacements.

Debye-Waller displacement parameters U (mean-square, in A^2) per species:
  - Cs: U = 0.05 A^2  -> RMS = sqrt(0.05) = 0.224 A
    Cs is the lightest cation and sits in the largest cage; weakest binding.
  - Pb: U = 0.02 A^2  -> RMS = sqrt(0.02) = 0.141 A
    Pb is heavy and forms the PbBr6 octahedra; moderate binding.
  - Br: U = 0.03 A^2  -> RMS = sqrt(0.03) = 0.173 A
    Br is intermediate in mass and binding strength.

Random strain: uniform random strain tensor with components in [-0.01, 0.01]
  (i.e. up to 1% strain in any direction), applied to the cell vectors.

Units: all displacements are in Angstroms (ASE convention).
"""

import numpy as np
import ase.io
from ase import Atoms
import random

# --- Parameters ---
T = 400  # K, cubic CsPbBr3 stable above ~300 K

# Debye-Waller U in A^2 (mean-square displacement per axis)
U_Cs = 0.05  # A^2
U_Pb = 0.02  # A^2
U_Br = 0.03  # A^2

# RMS displacement per axis = sqrt(U) in A
RMS_Cs = np.sqrt(U_Cs)  # ~0.224 A
RMS_Pb = np.sqrt(U_Pb)  # ~0.141 A
RMS_Br = np.sqrt(U_Br)  # ~0.173 A

# Strain: max component magnitude (fractional)
MAX_STRAIN = 0.01  # 1%

N_STRUCTURES = 51  # 000 (unperturbed) + 001..050 (perturbed)

# --- Read base structure ---
base = ase.io.read('structures/base_primitive.cif')
print(f"Base structure: {len(base)} atoms, species: {base.get_chemical_symbols()}")
print(f"Base cell: {base.cell[:].tolist()}")

# INVARIANT 1: nat and ntyp computed from the actual Atoms object
nat = len(base)
typ = len(set(base.get_chemical_symbols()))
print(f"nat = {nat}, ntyp = {typ}")
assert nat == 5, f"Expected 5 atoms in primitive CsPbBr3, got {nat}"
assert typ == 3, f"Expected 3 species, got {typ}"

# --- Write structure 000 (unperturbed) ---
ase.io.write('structures/structure_000.cif', base)
print("Wrote structures/structure_000.cif (unperturbed)")

# --- Generate perturbed structures 001-050 ---
random.seed(42)  # reproducibility
np.random.seed(42)

for i in range(1, N_STRUCTURES):
    atoms = base.copy()
    
    # 1. Apply small random lattice strain to cell vectors
    #    strain tensor: symmetric, components in [-MAX_STRAIN, MAX_STRAIN]
    strain = np.random.uniform(-MAX_STRAIN, MAX_STRAIN, (3, 3))
    # Make it symmetric (physical strain is symmetric)
    strain = 0.5 * (strain + strain.T)
    # New cell = (I + strain) @ old_cell
    new_cell = (np.eye(3) + strain) @ atoms.cell[:]
    atoms.set_cell(new_cell, scale_atoms=True)  # scale_atoms keeps fractional coords
    
    # 2. Apply random thermal displacements in Cartesian coordinates
    #    Each atom gets a Gaussian displacement with sigma = RMS for its species
    positions = atoms.get_positions().copy()
    for j, sym in enumerate(atoms.get_chemical_symbols()):
        if sym == 'Cs':
            sigma = RMS_Cs
        elif sym == 'Pb':
            sigma = RMS_Pb
        elif sym == 'Br':
            sigma = RMS_Br
        else:
            raise ValueError(f"Unknown species {sym}")
        positions[j] += np.random.normal(0, sigma, 3)
    atoms.set_positions(positions)
    
    # INVARIANT 1: verify nat/ntyp unchanged after perturbation
    assert len(atoms) == nat, f"Structure {i:03d}: atom count changed"
    assert len(set(atoms.get_chemical_symbols())) == typ, f"Structure {i:03d}: species count changed"
    
    fname = f'structures/structure_{i:03d}.cif'
    ase.io.write(fname, atoms)
    
    if i <= 3 or i == N_STRUCTURES - 1:
        print(f"Wrote {fname} (strain max={np.abs(strain).max():.4f}, "
              f"displacement range: {np.abs(positions - base.get_positions()).max():.4f} A)")

print(f"\nDone. Wrote {N_STRUCTURES} structures: structure_000.cif .. structure_{N_STRUCTURES-1:03d}.cif")
