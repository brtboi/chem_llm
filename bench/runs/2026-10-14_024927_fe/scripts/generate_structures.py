#!/usr/bin/env python
"""
STEP 2: Generate 51 perturbed bcc Fe structures (structure_000.cif .. structure_050.cif).

- structure_000.cif: unperturbed PRIMITIVE cell (reduced from the MP CIF via chem_llm.kpath).
- structure_001..050: random thermal displacements + small random lattice strain.

Temperature choice: T = 300 K.
  bcc Fe (alpha-Fe, Im-3m #229) is the stable phase at room temperature, stable up to the
  alpha->gamma transition at ~1185 K. 300 K is therefore a physically valid temperature at
  which this phase exists and thermal motion is meaningful.

Displacement model (Einstein model, per species):
  U (crystallographic displacement parameter) = mean-square displacement in A^2:
      U = 3 k_B T / (4 pi^2 m omega^2)  =  3 k_B T / (4 pi^2 m (2 pi f)^2 / (2 pi)^2)
  Simplified: U = 3 k_B T / (4 pi^2 m omega^2) where omega is the Einstein angular frequency.
  For Fe: m = 55.845 u, omega ~ 2 pi * 10^13 rad/s (typical Debye/Einstein frequency for Fe).
  Per-axis RMS displacement = sqrt(U) in A. We draw Gaussian displacements with
  std = sqrt(U) per Cartesian axis (isotropic Einstein model).

  U_Fe = 3 * k_B * T / (4 * pi^2 * m_Fe * omega^2)
  k_B = 1.380649e-23 J/K, m_Fe = 55.845 * 1.660539e-27 kg, omega = 2*pi*1e13 rad/s
  U_Fe ~ 3*1.380649e-23*300 / (4*pi^2 * 55.845*1.660539e-27 * (2*pi*1e13)^2)
       ~ 1.2426e-20 / (4*9.8696 * 9.272e-26 * 3.9478e27)
       ~ 1.2426e-20 / (1.4534e3)  ~ 8.55e-24 m^2 = 8.55e-8 A^2
  RMS = sqrt(8.55e-8) A ~ 2.9e-4 A  (very small, as expected for a heavy atom at 300 K)

  For a more visible perturbation representative of thermal motion in a DFT context,
  we use a slightly larger effective amplitude: RMS = 0.01 A per axis (10 pm), which is
  ~30x the Einstein estimate but still << lattice constant (2.48 A) and representative of
  the kind of atomic jitter one would sample in an ab-initio MD trajectory at 300 K.
  This is a deliberate choice to make the perturbation numerically meaningful for testing
  the workflow while remaining physically plausible.

Lattice strain: small random symmetric strain tensor with components ~ N(0, 0.001) (0.1%).

Units summary:
  - U: A^2 (mean-square displacement)
  - RMS per axis: sqrt(U) in A
  - Displacements drawn: Gaussian, std = 0.01 A per Cartesian axis
  - Strain: dimensionless (delta_L/L ~ 0.001)
"""

import numpy as np
from ase.io import read, write
from ase import Atoms
from chem_llm.kpath import standard_kpath, kp_atoms

# --- Read base CIF (mp-13, bcc Fe, 2 atoms in doubled/oblique setting) ---
base = read('structures/structure_000.cif')
print(f"Base CIF: {len(base)} atoms, cell = {base.cell[:3]}")

# --- Reduce to primitive cell via chem_llm.kpath ---
# standard_kpath(atoms) returns the seekpath standard k-path for THIS structure;
# kp_atoms(kp) returns the primitive cell the path refers to.
# For bcc Fe: 2 atoms -> 1 atom (primitive bcc).
kp = standard_kpath(base)
prim = kp_atoms(kp)
print(f"Primitive cell: {len(prim)} atoms, space group: {kp['spacegroup']}")

# INVARIANT: space group number unchanged by reduction
sg_num_orig = 229  # from generate_cif
sg_num_prim = int(kp['spacegroup'].split('#')[-1].strip().rstrip(')'))
assert sg_num_prim == sg_num_orig, f"Space group changed: {sg_num_orig} -> {sg_num_prim}"

# INVARIANT: reduced atom count divides original evenly
assert len(base) % len(prim) == 0, f"{len(base)} not divisible by {len(prim)}"

# --- Write structure_000.cif: unperturbed primitive cell ---
# Preserve the FM initial magnetic moment on Fe (needed for spin-polarized runs)
prim.set_initial_magnetic_moments([0.5])  # 0.5 muB/atom initial guess for Fe (FM)
write('structures/structure_000.cif', prim)
print(f"Wrote structures/structure_000.cif ({len(prim)} atoms)")

# --- Thermal displacement parameters ---
# T = 300 K (alpha-Fe stable at RT, up to ~1185 K)
T = 300.0  # K
# Per-species RMS displacement (A):
# Fe: heavy atom, stiff bonds -> small displacement. RMS = 0.01 A per axis.
# (Einstein model gives ~0.0003 A; we use 0.01 A as a representative thermal jitter
#  that is numerically meaningful for DFT perturbation testing.)
RMS_Fe = 0.01  # A, per Cartesian axis (std of Gaussian displacement)

# Lattice strain: max component ~0.1% (0.001)
STRAIN_SIGMA = 0.001  # dimensionless

rng = np.random.default_rng(seed=42)

for i in range(1, 51):
    tag = f"{i:03d}"
    atoms = prim.copy()

    # --- Random lattice strain ---
    # Apply a small random symmetric strain tensor to the cell.
    # strain = [[e11, e12, e13], [e12, e22, e23], [e13, e23, e33]]
    # New cell = (I + strain) @ old_cell
    strain = rng.normal(0.0, STRAIN_SIGMA, size=(3, 3))
    strain = 0.5 * (strain + strain.T)  # symmetrize
    new_cell = (np.eye(3) + strain) @ np.array(atoms.cell[:3])
    atoms.cell = new_cell

    # --- Random atomic displacements (thermal motion) ---
    # Displace each atom by a Gaussian random vector with std = RMS_Fe per axis.
    # Displacements are in Cartesian coordinates (A).
    disp = rng.normal(0.0, RMS_Fe, size=(len(atoms), 3))
    atoms.positions += disp

    # --- INVARIANT 1: nat/ntyp check ---
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 1, f"structure_{tag}: expected nat=1, got {nat}"
    assert ntyp == 1, f"structure_{tag}: expected ntyp=1, got {ntyp}"

    # Preserve initial magnetic moment
    atoms.set_initial_magnetic_moments([0.5])

    write(f'structures/structure_{tag}.cif', atoms)

print("Done: wrote structures/structure_000.cif through structure_050.cif")
