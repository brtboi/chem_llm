import numpy as np
import ase.io
from ase import Atoms
from chem_llm.kpath import standard_kpath, kp_atoms

# =============================================================================
# generate_structures.py
# Generates 51 perturbed fcc Ni structures (structure_000.cif ... structure_050.cif)
# structure_000 = unperturbed primitive cell
# structures 001-050 = random atomic displacements + small random lattice strain
# =============================================================================

# --- Temperature choice ---
# fcc Ni is the stable phase from 0 K up to the melting point (~1728 K).
# We choose T = 300 K (room temperature), well within the fcc stability range.
# This is representative of ambient thermal motion.
T = 300  # Kelvin

# --- Displacement amplitudes ---
# For a monatomic fcc metal, the mean-square displacement U (in A^2) is
# estimated from the Debye-Waller factor. For Ni at 300 K, the Debye temperature
# is ~480 K, giving a typical RMS displacement of ~0.01 A per axis.
# We use a slightly larger value (0.02 A RMS per axis) to ensure visible
# perturbation while remaining physically reasonable for thermal motion.
# U = <u^2> = (0.02 A)^2 = 4e-4 A^2  (mean-square displacement, A^2)
# RMS per axis = sqrt(U) = 0.02 A
RMS_DISPLACEMENT = 0.02  # A, per-axis RMS displacement (sqrt of U)

# --- Lattice strain ---
# Small random strain: up to 0.5% in each direction, representing
# thermal expansion fluctuations and small defects.
MAX_STRAIN = 0.005  # 0.5% maximum strain per axis

# --- Seed for reproducibility ---
np.random.seed(42)

# --- Read base structure and reduce to primitive cell ---
base = ase.io.read('structures/base_Ni.cif')
print(f"Base structure: {len(base)} atoms, symbols: {base.get_chemical_symbols()}")

kp = standard_kpath(base)
prim = kp_atoms(kp)
print(f"Primitive cell: {len(prim)} atoms, space group: {kp['spacegroup']}")

# INVARIANT 1: nat/ntyp computed from actual Atoms object
nat = len(prim)
types = set(prim.get_chemical_symbols())
ntyp = len(types)
print(f"nat = {nat}, ntyp = {ntyp}, types = {types}")
assert nat == 1, f"Expected 1 atom in primitive fcc Ni cell, got {nat}"
assert ntyp == 1, f"Expected 1 type, got {ntyp}"

# --- Write structure_000 (unperturbed) ---
ase.io.write('structures/structure_000.cif', prim)
print("Wrote structures/structure_000.cif (unperturbed)")

# --- Generate perturbed structures ---
for i in range(1, 51):
    # Start from a fresh copy of the primitive cell
    atoms = prim.copy()

    # 1) Random atomic displacements (thermal motion)
    #    Displacements are Gaussian with sigma = RMS_DISPLACEMENT per axis
    #    Units: Angstrom
    disp = np.random.normal(0.0, RMS_DISPLACEMENT, size=(nat, 3))
    atoms.positions += disp

    # 2) Small random lattice strain
    #    Apply a diagonal strain tensor: new_cell = (1 + strain) * old_cell
    #    Strain is in percent (dimensionless), up to MAX_STRAIN
    strain = np.random.uniform(-MAX_STRAIN, MAX_STRAIN, size=3)
    cell = atoms.cell.copy()
    # Apply strain to each cell vector independently
    cell[0] *= (1.0 + strain[0])
    cell[1] *= (1.0 + strain[1])
    cell[2] *= (1.0 + strain[2])
    atoms.cell = cell

    # INVARIANT 1: verify nat/ntyp for each structure
    assert len(atoms) == nat, f"Structure {i:03d}: nat changed to {len(atoms)}"
    assert len(set(atoms.get_chemical_symbols())) == ntyp, \
        f"Structure {i:03d}: ntyp changed"

    # Write the perturbed structure
    fname = f'structures/structure_{i:03d}.cif'
    ase.io.write(fname, atoms)
    print(f"Wrote {fname} (strain={strain}, max_disp={np.max(np.abs(disp)):.4f} A)")

print("\nDone. Generated 51 structures (000-050).")
