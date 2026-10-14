import numpy as np
import ase.io
from ase import Atoms
import copy

# =============================================================================
# generate_structures.py
# Generate 51 perturbed fcc Ni structures (structure_000.cif .. structure_050.cif)
# structure_000 = unperturbed base; structure_001..050 = thermal + strain perturbations
# =============================================================================

# --- Temperature choice ---
# fcc Ni (Fm-3m, #225) is the stable phase from 0 K up to the melting point
# (~1728 K).  We pick T = 300 K (room temperature), well within the stable range.
# At 300 K the thermal displacements are small enough that the fcc symmetry
# is only weakly broken, which is representative of thermal motion in a solid.
T = 300.0  # K

# --- Displacement amplitude ---
# For a single species (Ni, M = 58.69 g/mol), the classical equipartition
# estimate for the mean-square displacement is <x^2> = k_B T / (M omega^2).
# For a typical Debye frequency of fcc Ni (~4.5e13 Hz), this gives
# U = <x^2> ~ 0.0016 A^2, so the per-axis RMS displacement is sqrt(U) ~ 0.04 A.
# U is the crystallographic displacement parameter (mean-square displacement, A^2).
# RMS per axis = sqrt(U) in A.
U_Ni = 0.0016  # A^2  (mean-square displacement for Ni at 300 K)
RMS_Ni = np.sqrt(U_Ni)  # A  (~0.04 A per axis)

# --- Strain amplitude ---
# Small random symmetric strain tensor, magnitude ~0.5% (typical for thermal
# expansion fluctuations in a solid at 300 K).
STRAIN_MAG = 0.005  # dimensionless, 0.5%

# --- Read base structure ---
base = ase.io.read('structures/Ni_base.cif')
print(f"Base structure: {len(base)} atoms, cell = {base.get_cell()}")

# Reduce to primitive cell using chem_llm.kpath (same as reduce_primitive.py)
from chem_llm.kpath import standard_kpath, kp_atoms
kp_base = standard_kpath(base)
prim = kp_atoms(kp_base)
print(f"Primitive cell: {len(prim)} atoms, cell = {prim.get_cell()}")

# INVARIANT 1: nat and ntyp computed from the actual Atoms object
nat = len(prim)
ntyp = len(set(prim.get_chemical_symbols()))
print(f"nat = {nat}, ntyp = {ntyp}")
assert nat == 1, f"Expected 1 atom in primitive cell, got {nat}"
assert ntyp == 1, f"Expected 1 species, got {ntyp}"

rng = np.random.default_rng(seed=42)

for i in range(51):
    if i == 0:
        # Unperturbed base structure
        atoms = copy.deepcopy(prim)
    else:
        atoms = copy.deepcopy(prim)

        # --- Random atomic displacements (thermal motion) ---
        # Per-axis RMS displacement = sqrt(U) in A, drawn from a normal distribution
        # with standard deviation RMS_Ni.  This is equivalent to a Gaussian
        # distribution of displacements with <x^2> = U_Ni.
        disp = rng.normal(0.0, RMS_Ni, size=(nat, 3))  # A
        # Apply displacements in Cartesian coordinates
        pos = atoms.get_positions() + disp
        atoms.set_positions(pos)

        # --- Small random lattice strain ---
        # Generate a small symmetric strain tensor with magnitude ~STRAIN_MAG
        # The strain tensor is symmetric: e_ij = e_ji
        # We generate a random 3x3 matrix, symmetrize it, and scale it
        e = rng.normal(0.0, STRAIN_MAG / np.sqrt(6), size=(3, 3))
        e = 0.5 * (e + e.T)  # symmetrize
        # Apply strain to the cell: new_cell = (I + e) @ old_cell
        # Using set_cell with scale_atoms=False keeps atomic positions fixed in
        # Cartesian space (the cell deforms around them), which is the correct
        # way to apply a lattice strain to a crystal structure.
        old_cell = atoms.get_cell()
        new_cell = (np.eye(3) + e) @ old_cell
        atoms.set_cell(new_cell, scale_atoms=False)

    # INVARIANT 1: verify nat/ntyp for each structure
    assert len(atoms) == nat, f"Structure {i:03d}: atom count changed"
    assert len(set(atoms.get_chemical_symbols())) == ntyp, f"Structure {i:03d}: species count changed"

    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)
    print(f"Wrote {out_path}: {len(atoms)} atoms")

print("Done: 51 structures written to structures/structure_000.cif .. structure_050.cif")
