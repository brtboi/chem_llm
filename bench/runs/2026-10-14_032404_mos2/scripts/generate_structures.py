import numpy as np
import ase.io
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

# =============================================================================
# generate_structures.py
# Generates 51 MoS2 2H structures: structure_000.cif (unperturbed) through
# structure_050.cif (perturbed with thermal displacements + random strain).
# =============================================================================

# Temperature choice: 300 K (room temperature).
# 2H-MoS2 is the thermodynamically stable phase at room temperature and below.
# The 1T phase is only stable at high temperatures (>~1000 K) or under specific
# chemical potential conditions. At 300 K, 2H is the ground-state phase, so
# thermal displacements at this temperature are physically meaningful.
T = 300.0  # K

# Species-specific displacement amplitudes.
# U is the mean-square displacement in A^2 (crystallographic Debye-Waller factor).
# The per-axis RMS displacement is sqrt(U) in A.
# S atoms are lighter (32 amu) and more weakly bound than Mo (96 amu), so they
# have larger displacements. Typical U values for MoS2 at 300 K from literature:
#   Mo: U ~ 0.003 A^2  -> RMS ~ 0.055 A
#   S:  U ~ 0.006 A^2  -> RMS ~ 0.077 A
# We use these as the per-axis RMS displacements (sqrt(U)).
U_Mo = 0.003  # A^2, mean-square displacement for Mo
U_S = 0.006   # A^2, mean-square displacement for S
rms_Mo = np.sqrt(U_Mo)  # A, per-axis RMS displacement
rms_S = np.sqrt(U_S)    # A, per-axis RMS displacement

# Random strain: small isotropic + anisotropic strain, magnitude ~0.5%
# This represents lattice imperfections and thermal expansion effects.
strain_max = 0.005  # 0.5% maximum strain component

np.random.seed(42)  # reproducibility

# Read the primitive cell structure
base = ase.io.read('structures/MoS2_primitive.cif')
print(f"Base structure: {len(base)} atoms, cell = {base.get_cell()}")
print(f"Space group check: {base.get_chemical_symbols()}")

# Write structure_000.cif (unperturbed base)
ase.io.write('structures/structure_000.cif', base)
print("Wrote structures/structure_000.cif (unperturbed)")

for i in range(1, 51):
    atoms = base.copy()
    
    # Apply random atomic displacements (thermal motion)
    # Each atom gets a Gaussian displacement with species-specific RMS amplitude
    displacements = np.zeros((len(atoms), 3))
    for idx, symbol in enumerate(atoms.get_chemical_symbols()):
        if symbol == 'Mo':
            rms = rms_Mo
        elif symbol == 'S':
            rms = rms_S
        else:
            rms = 0.05  # fallback
        displacements[idx] = np.random.normal(0.0, rms, 3)
    
    atoms.positions += displacements
    
    # Apply small random lattice strain
    # Strain tensor: symmetric, traceless (volume-preserving) + small isotropic part
    strain = np.random.uniform(-strain_max, strain_max, (3, 3))
    strain = (strain + strain.T) / 2.0  # make symmetric
    # Remove trace (volume change) to keep it purely shear + small isotropic
    trace = np.trace(strain) / 3.0
    strain -= trace * np.eye(3)
    # Add a small isotropic component (thermal expansion ~0.1%)
    strain += np.random.uniform(-0.001, 0.001) * np.eye(3)
    
    # Apply strain to the cell
    # new_cell = (I + strain) @ old_cell
    old_cell = np.array(atoms.get_cell())
    new_cell = (np.eye(3) + strain) @ old_cell
    atoms.set_cell(new_cell, scale_atoms=True)
    
    # INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 6, f"Structure {i:03d}: expected 6 atoms, got {nat}"
    assert ntyp == 2, f"Structure {i:03d}: expected 2 species, got {ntyp}"
    
    # Write the perturbed structure
    fname = f'structures/structure_{i:03d}.cif'
    ase.io.write(fname, atoms)
    print(f"Wrote {fname} (nat={nat}, ntyp={ntyp})")

print("\nDone: 51 structures written (structure_000.cif through structure_050.cif)")
