import numpy as np
import ase.io
from ase import Atoms

# =============================================================================
# STEP 2: Generate 51 perturbed MoS2 structures (000 = unperturbed base)
# =============================================================================
#
# Temperature choice: 300 K (room temperature).
#   2H-MoS2 is the thermodynamically stable phase at ambient conditions;
#   the 1T phase is only stable at high temperatures (>~1000 K) or under
#   specific chemical potentials. 300 K is well within the stability range
#   of 2H-MoS2, so thermal displacements at this temperature are physically
#   meaningful for this phase.
#
# Displacement amplitudes (per-axis RMS, in Angstrom):
#   These are mean-square displacement (U) values converted to RMS:
#   RMS = sqrt(U), where U is the crystallographic displacement parameter
#   in A^2 (mean-square displacement).
#
#   For MoS2 at 300 K, typical Debye-Waller factors from X-ray/neutron
#   diffraction give:
#     Mo: U ~ 0.003 A^2  -> RMS = sqrt(0.003) ~ 0.055 A
#     S:  U ~ 0.005 A^2  -> RMS = sqrt(0.005) ~ 0.071 A
#
#   Sulfur is lighter (32 amu vs 96 amu for Mo) and more weakly bound
#   (terminal atoms in the S-Mo-S sandwich), so it moves more. This is
#   consistent with the larger U for S.
#
# Lattice strain: small random strain, ~0.1% per component.
#   This represents the small lattice distortions that accompany thermal
#   expansion and local strain fields. 0.1% is well within the elastic
#   regime for MoS2 (bulk modulus ~ 150 GPa, so 0.1% strain ~ 0.15 GPa
#   stress, far below the yield strength).
#
# =============================================================================

# Read the primitive base structure (already reduced in STEP 1)
base = ase.io.read('structures/MoS2_primitive.cif')
print(f'Base structure: {len(base)} atoms, symbols = {base.get_chemical_symbols()}')

# Per-species RMS displacement amplitudes (in Angstrom)
# Derived from typical Debye-Waller factors at 300 K:
#   Mo: U = 0.003 A^2 -> RMS = sqrt(0.003) = 0.0548 A
#   S:  U = 0.005 A^2 -> RMS = sqrt(0.005) = 0.0707 A
rms_displacements = {
    'Mo': np.sqrt(0.003),  # ~0.055 A, heavier atom, moves less
    'S':  np.sqrt(0.005),  # ~0.071 A, lighter atom, moves more
}

# Random strain amplitude (fractional, per component)
# 0.1% = 0.001, applied as a random 3x3 strain tensor
strain_amplitude = 0.001

# Number of structures to generate
n_structures = 51

# Fixed seed for reproducibility
np.random.seed(42)

for i in range(n_structures):
    if i == 0:
        # Structure 000: unperturbed base
        atoms = base.copy()
    else:
        # Perturbed copy: random atomic displacements + small random lattice strain
        atoms = base.copy()

        # Apply random lattice strain to the cell
        # Strain tensor: small random values, symmetric part only
        strain = np.random.randn(3, 3) * strain_amplitude
        # Make it symmetric (physical strain is symmetric)
        strain = 0.5 * (strain + strain.T)
        # New cell = (I + strain) @ old_cell
        new_cell = (np.eye(3) + strain) @ base.cell[:]
        atoms.set_cell(new_cell, scale_atoms=True)

        # Apply random atomic displacements (in Cartesian coordinates, Angstrom)
        # Each atom gets a Gaussian displacement with its species-specific RMS
        for j, sym in enumerate(atoms.get_chemical_symbols()):
            disp = np.random.randn(3) * rms_displacements[sym]
            atoms.positions[j] += disp

    # INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 6, f'Structure {i:03d}: expected 6 atoms, got {nat}'
    assert ntyp == 2, f'Structure {i:03d}: expected 2 species, got {ntyp}'

    # Write the structure
    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)

    if i < 3 or i == n_structures - 1:
        print(f'Wrote {out_path}: {nat} atoms, {ntyp} species')

print(f'\nDone: wrote {n_structures} structures (000-050)')
print(f'  000: unperturbed base')
print(f'  001-050: perturbed (T=300K, Mo RMS={rms_displacements["Mo"]:.4f} A, '
      f'S RMS={rms_displacements["S"]:.4f} A, strain~{strain_amplitude*100:.1f}%)')
