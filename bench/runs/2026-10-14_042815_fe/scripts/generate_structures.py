import numpy as np
import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

# =============================================================================
# generate_structures.py
# Writes 51 structures: structure_000.cif (unperturbed primitive bcc Fe) and
# structure_001..050.cif (thermal displacements + small random lattice strain).
# =============================================================================

# --- Temperature choice -------------------------------------------------------
# bcc Fe (alpha-Fe) is the stable phase below the alpha->gamma (bcc->fcc)
# transformation at T_c = 1185 K. We choose T = 1000 K: well below T_c so the
# bcc phase is thermodynamically stable, yet high enough that thermal motion is
# significant (Debye temperature of Fe ~ 470 K, so kT >> k*Theta_D/3).
T = 1000.0  # Kelvin

# --- Displacement amplitudes --------------------------------------------------
# Crystallographic displacement parameters U are mean-square displacements in A^2.
# The per-axis RMS displacement is sqrt(U) in A.
# For a single-element system (Fe) we use the Einstein model:
#   U = (3 * hbar^2) / (4 * pi^2 * m * c^2 * T)  ... (simplified harmonic approx)
# A more practical approach: use the Debye model at high T (T >> Theta_D):
#   <u^2> = 3 * hbar^2 / (m * k_B * T)  (per atom, 3D total)
# Per-axis: U_axis = <u^2> / 3 = hbar^2 / (m * k_B * T)
# For Fe: m = 55.845 u = 9.274e-26 kg
#   U_axis = (1.0546e-34)^2 / (9.274e-26 * 1.3806e-23 * 1000)
#          = 1.112e-68 / 1.280e-45 = 8.68e-24 m^2 = 8.68e-16 A^2
# That's the quantum limit. At T=1000K >> Theta_D=470K, the classical limit applies:
#   <u^2> = 3 * k_B * T / (m * omega^2)  where omega ~ Debye frequency
# Debye frequency: omega_D = k_B * Theta_D / hbar = 1.3806e-23 * 470 / 1.0546e-34 = 6.16e13 rad/s
# Classical per-axis: U_axis = k_B * T / (m * omega_D^2)
#   = 1.3806e-23 * 1000 / (9.274e-26 * (6.16e13)^2)
#   = 1.3806e-20 / (9.274e-26 * 3.79e27)
#   = 1.3806e-20 / 3.515e2 = 3.93e-23 m^2 = 3.93e-15 A^2
# sqrt(U_axis) = 6.3e-8 m = 0.063 A  -- this is the per-axis RMS displacement.
#
# In practice, for DFT perturbation studies, displacements of ~0.05-0.1 A are
# typical for thermal motion at 1000 K. We use:
#   U_Fe = 0.0025 A^2  (mean-square displacement per axis)
#   RMS per axis = sqrt(0.0025) = 0.05 A
# This is a reasonable estimate for Fe at 1000 K (slightly below the classical
# Debye estimate of 0.063 A, accounting for the fact that the actual phonon
# spectrum has lower-frequency modes that contribute less).
# Since there is only one species (Fe), all atoms have the same amplitude.
U_Fe = 0.0025  # A^2, mean-square displacement per axis
RMS_Fe = np.sqrt(U_Fe)  # A, per-axis RMS displacement = 0.05 A

# --- Lattice strain -----------------------------------------------------------
# Small random homogeneous strain: |epsilon_ij| <= 0.01 (1%)
# This represents small deviations from the ideal lattice constant due to
# thermal expansion and local strain fields.
MAX_STRAIN = 0.01  # dimensionless, 1%

# --- Read base structure ------------------------------------------------------
# The base is the primitive cell (1 atom) written by reduce_cell.py
atoms_base = ase.io.read('structures/Fe_primitive.cif')
print(f'Base structure: {len(atoms_base)} atoms, cell = {atoms_base.get_cell()}')

# Verify it's the primitive cell
kp = standard_kpath(atoms_base)
prim = kp_atoms(kp)
assert len(prim) == 1, f'Expected 1 atom in primitive cell, got {len(prim)}'
assert '229' in kp['spacegroup'], f'Expected space group #229, got {kp["spacegroup"]}'

# --- Generate structures ------------------------------------------------------
np.random.seed(42)  # reproducibility

for i in range(51):
    if i == 0:
        # Unperturbed base structure
        atoms = ase.Atoms(
            symbols=atoms_base.get_chemical_symbols(),
            positions=atoms_base.get_positions(),
            cell=atoms_base.get_cell(),
            pbc=True,
        )
    else:
        # Perturbed copy: thermal displacements + random lattice strain
        atoms = atoms_base.copy()
        
        # 1) Random lattice strain: apply a small symmetric strain tensor
        #    epsilon_ij = random in [-MAX_STRAIN, MAX_STRAIN], symmetric
        eps = np.random.uniform(-MAX_STRAIN, MAX_STRAIN, size=(3, 3))
        eps = 0.5 * (eps + eps.T)  # make symmetric
        # New cell = (I + eps) @ old_cell  (strain in Cartesian coordinates)
        old_cell = atoms.get_cell()
        new_cell = (np.eye(3) + eps) @ old_cell.array
        atoms.set_cell(new_cell, scale_atoms=True)  # scale positions with cell
        
        # 2) Thermal displacements: Gaussian, per-axis RMS = RMS_Fe
        #    Displacements are in Cartesian coordinates (A)
        displacements = np.random.normal(0.0, RMS_Fe, size=atoms.get_positions().shape)
        atoms.positions += displacements
    
    # INVARIANT 1: nat and ntyp computed from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 1, f'structure_{i:03d}: expected nat=1, got {nat}'
    assert ntyp == 1, f'structure_{i:03d}: expected ntyp=1, got {ntyp}'
    
    # Write CIF
    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)
    
    if i < 3 or i == 50:
        print(f'  {out_path}: nat={nat}, ntyp={ntyp}, ' +
              f'cell diag = {np.diag(atoms.get_cell().array).round(4)}')

print(f'\nDone: wrote 51 structures (structure_000.cif .. structure_050.cif)')
print(f'Temperature: {T} K (bcc Fe stable below 1185 K)')
print(f'U_Fe = {U_Fe} A^2 (mean-square per axis), RMS = {RMS_Fe:.4f} A')
print(f'Max strain: {MAX_STRAIN} (1%)')
