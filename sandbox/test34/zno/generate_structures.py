import ase.io
import numpy as np

# =============================================================================
# generate_structures.py
# Generates 51 wurtzite ZnO structures: structure_000.cif (unperturbed base)
# and structure_001.cif .. structure_050.cif (perturbed copies with thermal
# displacements + small random lattice strain).
#
# Temperature choice: 300 K (room temperature).
# Wurtzite ZnO is the thermodynamically stable phase at ambient conditions.
# The competing zincblende (sphalerite) phase is only ~1-2 kJ/mol higher in
# energy at 0 K and does not become the stable phase until very high
# pressures (>10 GPa) or temperatures well above 1000 K. At 300 K wurtzite
# is unambiguously the ground-state structure, so 300 K is the correct
# temperature for thermal-displacement amplitudes.
#
# Displacement amplitudes (Einstein model):
#   RMS displacement per axis:  u_rms = sqrt(3 * k_B * T / M)
#   where k_B = 1.380649e-23 J/K, T = 300 K, M = atomic mass in kg.
#   The crystallographic mean-square displacement parameter is U = u_rms^2
#   (in A^2). We displace each atom by a Gaussian with standard deviation
#   u_rms (in A) along each Cartesian axis independently.
#
#   O  (Z=8,  M = 15.999 g/mol = 15.999e-3 kg):
#       u_rms = sqrt(3 * 1.380649e-23 * 300 / 15.999e-3) = sqrt(7.745e-19)
#             = 8.80e-10 m = 0.880 A
#       U_O   = u_rms^2 = 0.774 A^2
#
#   Zn (Z=30, M = 65.38 g/mol = 65.38e-3 kg):
#       u_rms = sqrt(3 * 1.380649e-23 * 300 / 65.38e-3) = sqrt(1.903e-19)
#             = 4.36e-10 m = 0.436 A
#       U_Zn  = u_rms^2 = 0.190 A^2
#
#   O moves ~2x more than Zn because it is ~4x lighter (u_rms ~ 1/sqrt(M)).
#
# Lattice strain: small random strain tensor with per-axis magnitude ~0.5%
# (i.e. strain components drawn uniformly from [-0.005, +0.005]). This is
# representative of the kind of lattice distortion that would arise from
# thermal expansion fluctuations or small defects, without changing the
# crystal structure type.
# =============================================================================

k_B = 1.380649e-23  # J/K
T = 300.0           # K

# Atomic masses in kg
M_O = 15.999e-3
M_Zn = 65.38e-3

# RMS displacement per axis in Angstrom (1 A = 1e-10 m)
# u_rms [A] = sqrt(3 * k_B * T / M) / 1e-10
u_rms_O = np.sqrt(3 * k_B * T / M_O) / 1e-10   # ~0.880 A
u_rms_Zn = np.sqrt(3 * k_B * T / M_Zn) / 1e-10  # ~0.436 A

# Mean-square displacement parameters U in A^2 (for reference / comments)
U_O = u_rms_O ** 2    # ~0.774 A^2
U_Zn = u_rms_Zn ** 2  # ~0.190 A^2

print(f"T = {T} K")
print(f"O:  u_rms = {u_rms_O:.4f} A,  U = {U_O:.4f} A^2")
print(f"Zn: u_rms = {u_rms_Zn:.4f} A,  U = {U_Zn:.4f} A^2")

# Read the base (unperturbed) structure
base = ase.io.read('structures/structure_000.cif')
print(f"Base structure: nat = {len(base)}, symbols = {base.get_chemical_symbols()}")

# INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
nat = len(base)
typ = len(set(base.get_chemical_symbols()))
print(f"nat = {nat}, ntyp = {typ}")

# Build a per-atom RMS displacement array
# Each atom gets its species-specific u_rms
u_rms_per_atom = np.array([
    u_rms_O if sym == 'O' else u_rms_Zn
    for sym in base.get_chemical_symbols()
])

rng = np.random.default_rng(seed=42)  # reproducible

# Structure 000: unperturbed base
ase.io.write('structures/structure_000.cif', base)
print("Wrote structures/structure_000.cif (unperturbed)")

for n in range(1, 51):
    atoms = base.copy()

    # --- Thermal displacements (Gaussian, per-axis, per-species amplitude) ---
    # Displace in Cartesian coordinates: each axis gets an independent
    # Gaussian with std = u_rms for that species.
    pos = atoms.get_positions().copy()  # Cartesian, Angstrom
    for i in range(nat):
        pos[i] += rng.normal(0.0, u_rms_per_atom[i], size=3)
    atoms.set_positions(pos)

    # --- Small random lattice strain ---
    # Strain tensor: symmetric, per-axis components ~ U[-0.005, 0.005]
    # This is a small perturbation (~0.5%) that does not change the structure type.
    strain = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            val = rng.uniform(-0.005, 0.005)
            strain[i, j] = val
            strain[j, i] = val  # symmetric strain tensor

    # Apply strain to the cell: new_cell = (I + strain) @ old_cell
    # In ASE, cell vectors are rows of cell.array.
    # If we think of the cell as a matrix C (rows = vectors),
    # a strain tensor epsilon transforms positions as x' = (I + eps) x,
    # so the new cell is C' = (I + eps) @ C  (matrix multiplication).
    old_cell = np.array(atoms.cell.array)  # 3x3, rows = cell vectors
    new_cell = (np.eye(3) + strain) @ old_cell
    atoms.set_cell(new_cell, scale_atoms=True)  # scale_atoms=True keeps fractional coords fixed

    # INVARIANT 1: verify nat/ntyp are unchanged after perturbation
    assert len(atoms) == nat, f"Structure {n:03d}: nat changed from {nat} to {len(atoms)}"
    assert len(set(atoms.get_chemical_symbols())) == typ, \
        f"Structure {n:03d}: ntyp changed from {typ} to {len(set(atoms.get_chemical_symbols()))}"

    fname = f'structures/structure_{n:03d}.cif'
    ase.io.write(fname, atoms)
    print(f"Wrote {fname} (nat={len(atoms)}, ntyp={len(set(atoms.get_chemical_symbols()))})")

print("\nDone: 51 structures written (structure_000.cif .. structure_050.cif)")
