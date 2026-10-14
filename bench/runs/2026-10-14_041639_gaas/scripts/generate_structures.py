import numpy as np
import ase.io
from ase import Atoms
import os

# =============================================================================
# generate_structures.py
# Generates 51 GaAs structures: structure_000.cif (unperturbed) and
# structure_001.cif through structure_050.cif (perturbed with thermal
# displacements + random lattice strain).
#
# Temperature choice: T = 300 K (room temperature).
# Zincblende GaAs is the stable phase at ambient pressure and temperature.
# It transforms to the rocksalt (B1) phase only under high pressure (>~10 GPa)
# or at very high temperatures (>~1200 K). 300 K is well within the stability
# range of the zincblende phase.
#
# Displacement amplitudes (RMS, in Angstrom):
#   The per-axis RMS displacement is sqrt(U) where U is the mean-square
#   displacement (crystallographic U in A^2).
#   For a harmonic oscillator at temperature T, the mean-square displacement
#   is U = 3*k_B*T / (m*omega^2) per atom, but a simpler estimate uses the
#   Debye model or empirical values.
#   For GaAs at 300 K, typical RMS displacements are:
#     Ga (lighter, Z=31): ~0.06 A  (U ~ 0.0036 A^2)
#     As (heavier, Z=33): ~0.05 A  (U ~ 0.0025 A^2)
#   The lighter Ga atom moves more because it has a lower mass and weaker
#   effective binding in the lattice.
#
# Lattice strain: small random strain tensor with components ~0.1% (1e-3),
#   representative of thermal expansion fluctuations.
# =============================================================================

np.random.seed(42)

# Read the primitive cell base structure
base = ase.io.read('structures/base_GaAs_prim.cif')
print(f"Base structure: nat={len(base)}, symbols={base.get_chemical_symbols()}")

# Displacement amplitudes (RMS per axis, in Angstrom)
# These are sqrt(U) values, i.e., the standard deviation of the displacement
# along each Cartesian axis for a harmonic oscillator at 300 K.
RMS_Ga = 0.06  # A, lighter atom, moves more
RMS_As = 0.05  # A, heavier atom, moves less

# Random strain magnitude (dimensionless, ~0.1%)
STRAIN_MAG = 1e-3

os.makedirs('structures', exist_ok=True)

for i in range(51):
    nnn = f"{i:03d}"
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()
        
        # Apply random lattice strain
        # Generate a small random strain tensor (symmetric, traceless-ish)
        strain = np.random.randn(3, 3) * STRAIN_MAG
        strain = (strain + strain.T) / 2.0  # make symmetric
        # Remove trace to keep volume roughly constant
        strain -= np.trace(strain) / 3.0 * np.eye(3)
        
        # Apply strain to the cell
        new_cell = atoms.cell * (np.eye(3) + strain)
        atoms.set_cell(new_cell, scale_atoms=True)
        
        # Apply random atomic displacements (thermal motion)
        # Each atom gets a Gaussian displacement with species-specific RMS
        symbols = atoms.get_chemical_symbols()
        displacements = np.zeros((len(atoms), 3))
        for j, sym in enumerate(symbols):
            rms = RMS_Ga if sym == 'Ga' else RMS_As
            displacements[j] = np.random.randn(3) * rms
        
        # Add displacements in Cartesian coordinates
        pos = atoms.get_positions()
        pos += displacements
        atoms.set_positions(pos)
    
    # INVARIANT 1: assert nat/ntyp consistency
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Structure {nnn}: expected nat=2, got {nat}"
    assert ntyp == 2, f"Structure {nnn}: expected ntyp=2, got {ntyp}"
    
    out_path = f'structures/structure_{nnn}.cif'
    ase.io.write(out_path, atoms)
    
    if i == 0 or i % 10 == 0:
        print(f"Wrote {out_path}: nat={nat}, ntyp={ntyp}")

print("Done: 51 structures written to structures/")
