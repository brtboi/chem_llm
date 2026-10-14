import numpy as np
import ase.io
from ase import Atoms

# =============================================================================
# generate_structures.py
# Writes 51 LiF structures: structure_000.cif (unperturbed primitive cell)
# and structure_001..050.cif (thermal displacements + small random strain).
#
# Temperature choice: T = 300 K.
#   LiF is stable at room temperature (melting point ~1121 K, Debye temp ~1000 K),
#   so 300 K is well within the stable phase range.
#
# Displacement amplitudes (Einstein model with phonon frequency):
#   U (crystallographic mean-square displacement) = 3*kB*T / (M * omega^2) in A^2
#   Per-axis RMS displacement = sqrt(U) in A
#   omega = 2*pi*c*nu, where nu is the phonon frequency in cm^-1.
#   For LiF, the optical phonon frequency is ~400 cm^-1 (typical for ionic crystals).
#   Li (M = 6.94 g/mol) moves more than F (M = 19.0 g/mol) because it is lighter.
#
# Strain: small random symmetric strain tensor, |eps| <= 0.01 (1%)
# =============================================================================

T = 300.0  # K

# Atomic masses in g/mol
masses = {'Li': 6.94, 'F': 19.0}

# Phonon frequency for LiF (optical mode, typical for ionic crystals)
# nu = 400 cm^-1
nu_cm = 400.0  # cm^-1
c_cm = 2.998e10  # cm/s
omega = 2.0 * np.pi * c_cm * nu_cm  # rad/s

# Boltzmann constant in J/K
kB_J = 1.380649e-23  # J/K

# Per-axis RMS displacement in Angstrom: sqrt(3*kB*T / (M * omega^2))
# M in kg/atom, omega in rad/s
rms = {}
for sp, m_gmol in masses.items():
    m_kg = m_gmol * 1e-3 / 6.02214076e23  # kg per atom
    U_m2 = 3.0 * kB_J * T / (m_kg * omega**2)  # m^2
    rms[sp] = np.sqrt(U_m2) * 1e10  # Angstrom
    print(f'{sp}: M = {m_gmol} g/mol, per-axis RMS = {rms[sp]:.4f} A, U = {rms[sp]**2:.4e} A^2')

# Sanity check: RMS should be ~0.01-0.1 A for ionic crystals at 300 K
for sp, r in rms.items():
    assert 0.001 < r < 0.5, f'{sp}: RMS {r:.4f} A is unphysical'

# Strain magnitude (dimensionless, 1%)
max_strain = 0.01

rng = np.random.default_rng(42)

base = ase.io.read('structures/structure_000.cif')
print(f'Base structure: nat = {len(base)}, ntyp = {len(set(base.get_chemical_symbols()))}')

for i in range(51):
    if i == 0:
        atoms = base.copy()
    else:
        atoms = base.copy()
        # Random atomic displacements (thermal motion)
        for idx, sp in enumerate(atoms.get_chemical_symbols()):
            disp = rng.normal(0.0, rms[sp], size=3)
            atoms.positions[idx] += disp
        # Small random symmetric strain
        eps = rng.uniform(-max_strain, max_strain, size=(3, 3))
        eps = 0.5 * (eps + eps.T)  # make symmetric
        # Apply strain to cell: new_cell = (I + eps) @ old_cell
        # (strain in Cartesian, cell vectors are rows)
        new_cell = (np.eye(3) + eps) @ atoms.cell.array
        atoms.cell = new_cell
        # Scale positions with the cell (strain moves atoms with the lattice)
        # positions = scaled @ cell, so scaled positions stay the same
        # but we need to update positions = scaled @ new_cell
        scaled = atoms.get_scaled_positions()
        atoms.positions = scaled @ new_cell

    # INVARIANT 1: nat/ntyp from the actual object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f'structure_{i:03d}: expected nat=2, got {nat}'
    assert ntyp == 2, f'structure_{i:03d}: expected ntyp=2, got {ntyp}'

    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)

print(f'Wrote 51 structures: structure_000.cif .. structure_050.cif')
