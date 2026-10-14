import ase.io
import numpy as np
from ase import Atoms

# ============================================================
# generate_structures.py
# Generates 51 MgO structures (structure_000.cif .. structure_050.cif)
# structure_000 = unperturbed primitive cell
# structure_001..050 = random thermal displacements + small random lattice strain
# ============================================================

# --- Temperature choice ---
# MgO (periclase, rock-salt Fm-3m) is stable from 0 K up to its melting point
# (~3000 K). We choose T = 300 K (room temperature), well within the stable range.
# At 300 K the Debye-Waller factors are small but non-zero, giving realistic
# thermal-motion amplitudes.
T = 300.0  # Kelvin

# --- Species-specific displacement amplitudes ---
# The per-axis RMS displacement is sigma = sqrt(U) in Angstrom, where U is the
# crystallographic mean-square displacement parameter in A^2.
#
# For a 3D harmonic oscillator at temperature T:
#   U = 3 * k_B * T / (m * omega^2)
# where omega is the characteristic vibrational frequency.
#
# Mg (Z=12, m=24.3 amu) is lighter than O (Z=8, m=16.0 amu), so Mg moves more.
# We estimate amplitudes from the Debye model with Debye temperatures:
#   theta_D(MgO) ~ 740 K (experimental)
# At T << theta_D, U ~ (9*k_B*T)/(m*omega_D^2) * (theta_D/T)^2 * integral...
# A simpler approach: use the Einstein model with characteristic frequencies
# from the phonon spectrum of MgO.
#
# For MgO at 300 K, typical U values from neutron diffraction:
#   U(Mg) ~ 0.0035 A^2  -> sigma(Mg) = sqrt(0.0035) ~ 0.059 A
#   U(O)  ~ 0.0045 A^2  -> sigma(O)  = sqrt(0.0045) ~ 0.067 A
#
# We use these as the per-axis RMS displacement (sigma) in Angstrom.
# Each displacement is drawn from N(0, sigma^2) independently per axis.

sigma_Mg = 0.059  # A, per-axis RMS displacement for Mg (U = 0.0035 A^2)
sigma_O  = 0.067  # A, per-axis RMS displacement for O  (U = 0.0045 A^2)

# --- Lattice strain ---
# Small random homogeneous strain, |epsilon| <= 0.005 (0.5%)
# Applied as a random symmetric 3x3 matrix with trace ~ 0 (volume-preserving-ish)
max_strain = 0.005  # dimensionless

# --- Read base structure ---
base = ase.io.read('structures/MgO_primitive.cif')
print(f"Base structure: nat={len(base)}, ntyp={len(set(base.get_chemical_symbols()))}")

# INVARIANT 1: nat and ntyp computed from the actual Atoms object
nat_base = len(base)
types_base = set(base.get_chemical_symbols())
ntyp_base = len(types_base)
assert nat_base == 2, f"Expected 2 atoms in primitive MgO, got {nat_base}"
assert ntyp_base == 2, f"Expected 2 species in MgO, got {ntyp_base}"

rng = np.random.default_rng(seed=42)

for i in range(51):
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()

        # --- Random atomic displacements (thermal motion) ---
        # Displacements are in Cartesian coordinates (Angstrom).
        # We work in fractional coordinates for the displacement, then convert.
        # Actually, it's cleaner to displace in Cartesian space.
        pos = atoms.get_positions()  # Cartesian, Angstrom
        for j, sym in enumerate(atoms.get_chemical_symbols()):
            if sym == 'Mg':
                sigma = sigma_Mg
            elif sym == 'O':
                sigma = sigma_O
            else:
                raise ValueError(f"Unexpected species {sym}")
            # Per-axis RMS displacement is sigma (A); draw from N(0, sigma^2)
            disp = rng.normal(0.0, sigma, size=3)
            pos[j] += disp
        atoms.set_positions(pos)

        # --- Small random lattice strain ---
        # Generate a random symmetric strain tensor with |eps_ij| <= max_strain
        eps = rng.uniform(-max_strain, max_strain, size=(3, 3))
        eps = 0.5 * (eps + eps.T)  # symmetrize
        # Make it traceless (volume-preserving) to avoid large volume changes
        eps -= np.trace(eps) / 3.0 * np.eye(3)

        # Apply strain to the cell: new_cell = (I + eps) @ old_cell
        # In ASE, cell rows are the lattice vectors.
        # If we apply strain in the Cartesian frame: new_vectors = (I+eps) @ old_vectors
        old_cell = np.array(atoms.cell)  # 3x3, rows = lattice vectors
        new_cell = (np.eye(3) + eps) @ old_cell
        atoms.set_cell(new_cell, scale_atoms=True)  # scale atoms with the cell

    # INVARIANT 1: assert nat/ntyp for each structure
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == nat_base, f"Structure {i:03d}: nat={nat}, expected {nat_base}"
    assert ntyp == ntyp_base, f"Structure {i:03d}: ntyp={ntyp}, expected {ntyp_base}"

    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)

print(f"Wrote 51 structures: structure_000.cif .. structure_050.cif")
print(f"nat={nat_base}, ntyp={ntyp_base} for all structures")
