"""
Generate 51 perturbed MoS2 structures (structure_000.cif through structure_050.cif).

structure_000.cif is the unperturbed base (primitive cell).
structure_001..050.cif have random atomic displacements + small random lattice strain.

Temperature choice: 300 K (room temperature).
2H-MoS2 is the thermodynamically stable phase at room temperature and up to ~1100 K.
The 1T phase is only stable at high temperatures (>1000 K) or under specific synthesis conditions.
Therefore 300 K is appropriate for 2H-MoS2.

Displacement amplitudes (RMS, in Angstrom):
  - S atoms: lighter (32 amu) and more weakly bound (terminal in the S-Mo-S sandwich).
    Using a Debye-Waller estimate: U ~ 3kT/(M*omega^2). For S in MoS2, typical U ~ 0.005 A^2
    at 300 K, so RMS = sqrt(U) ~ 0.07 A. We use 0.07 A as the per-axis RMS displacement.
  - Mo atoms: heavier (96 amu) and more strongly bound (center of the sandwich).
    Typical U ~ 0.002 A^2 at 300 K, so RMS = sqrt(U) ~ 0.045 A. We use 0.045 A.

Lattice strain: small random strain tensor with components up to 0.5% (0.005),
representing thermal expansion fluctuations.

Units: all displacements are in Angstrom (ASE convention).
"""

import numpy as np
import ase.io
from ase import Atoms

# Read the base primitive cell
base = ase.io.read('structures/structure_000_base.cif')
print(f"Base structure: {len(base)} atoms, symbols: {base.get_chemical_symbols()}")

# Species-specific RMS displacement amplitudes (in Angstrom)
# S: lighter, more weakly bound -> larger displacement
# Mo: heavier, more strongly bound -> smaller displacement
RMS_S = 0.07   # A, per-axis RMS displacement for S
RMS_Mo = 0.045 # A, per-axis RMS displacement for Mo

# Random strain magnitude (dimensionless, ~0.5%)
STRAIN_MAX = 0.005

np.random.seed(42)  # reproducibility

# Write structure_000.cif (unperturbed base)
ase.io.write('structures/structure_000.cif', base)
print("Wrote structures/structure_000.cif (unperturbed)")

for i in range(1, 51):
    atoms = base.copy()
    
    # Apply random lattice strain
    # Random symmetric strain tensor with components in [-STRAIN_MAX, STRAIN_MAX]
    strain = np.random.uniform(-STRAIN_MAX, STRAIN_MAX, size=(3, 3))
    strain = 0.5 * (strain + strain.T)  # make symmetric
    
    # Apply strain to the cell
    # new_cell = (I + strain) @ old_cell  (strain in Cartesian, cell in Angstrom)
    old_cell = np.array(atoms.cell)
    new_cell = (np.eye(3) + strain) @ old_cell
    atoms.cell = new_cell
    
    # Apply random atomic displacements (in Cartesian Angstrom)
    symbols = atoms.get_chemical_symbols()
    disp = np.zeros((len(atoms), 3))
    for idx, sym in enumerate(symbols):
        if sym == 'S':
            disp[idx] = np.random.normal(0, RMS_S, size=3)
        elif sym == 'Mo':
            disp[idx] = np.random.normal(0, RMS_Mo, size=3)
    
    # Add displacements to Cartesian positions
    pos = np.array(atoms.get_positions())
    atoms.positions = pos + disp
    
    # INVARIANT 1: nat and ntyp must match the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 6, f"Structure {i:03d}: expected nat=6, got {nat}"
    assert ntyp == 2, f"Structure {i:03d}: expected ntyp=2, got {ntyp}"
    
    fname = f'structures/structure_{i:03d}.cif'
    ase.io.write(fname, atoms)
    
print("Wrote structures/structure_001.cif through structure_050.cif (perturbed)")
print("DONE: 51 structures written.")
