import numpy as np
import ase.io
from ase import Atoms

# =============================================================================
# generate_structures.py
# Generates 51 MgO structures: structure_000.cif (unperturbed base) and
# structure_001..050.cif (thermal displacements + small random lattice strain).
#
# Temperature choice: 300 K
#   Periclase MgO (rock-salt, Fm-3m) is the stable phase at ambient conditions.
#   The high-temperature bcc CsCl-type (B2) phase only appears above ~3000 K,
#   so 300 K is appropriate for this phase.
#
# Displacement amplitudes (Einstein model):
#   U = k_B * T / (3 * pi^2 * k_spring)   [A^2]  (mean-square displacement)
#   RMS per axis = sqrt(U)                [A]
#   We use RMS per-axis displacements (sqrt(U)) in Angstrom.
#
#   For MgO, the force constants are roughly:
#     Mg: k_spring ~ 120 N/m  -> U_Mg = k_B*300/(3*pi^2*120) ~ 0.0038 A^2
#     O:  k_spring ~  80 N/m  -> U_O  = k_B*300/(3*pi^2*80)  ~ 0.0057 A^2
#   RMS per axis:
#     Mg: sqrt(0.0038) ~ 0.062 A
#     O:  sqrt(0.0057) ~ 0.075 A
#   O is lighter and more weakly bound, so it moves more than Mg.
#
# Lattice strain: small random strain tensor with components in [-0.01, 0.01]
#   (i.e. up to 1% strain per axis), representative of thermal expansion/contraction.
# =============================================================================

np.random.seed(42)

# Read the primitive base structure
base = ase.io.read('structures/prim_MgO.cif')
print(f"Base structure: {len(base)} atoms, symbols: {base.get_chemical_symbols()}")

# Per-species RMS displacement amplitudes in Angstrom (sqrt(U), U in A^2)
# Derived from Einstein model at 300 K (see comments above)
RMS_DISPLACEMENT = {
    'Mg': 0.062,  # A, RMS per axis = sqrt(U), U ~ 0.0038 A^2
    'O':  0.075,  # A, RMS per axis = sqrt(U), U ~ 0.0057 A^2
}

N_STRUCTURES = 51  # 000 (unperturbed) + 001..050 (perturbed)

for i in range(N_STRUCTURES):
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()

        # Apply random atomic displacements (thermal motion)
        # Displacements are in Cartesian coordinates (Angstrom)
        # Each atom gets a random Gaussian displacement with per-species RMS
        disp = np.zeros((len(atoms), 3))
        for j, sym in enumerate(atoms.get_chemical_symbols()):
            rms = RMS_DISPLACEMENT[sym]
            disp[j] = np.random.normal(0.0, rms, size=3)
        atoms.positions += disp

        # Apply small random lattice strain
        # Strain tensor components in [-0.01, 0.01] (up to 1% per axis)
        strain = np.random.uniform(-0.01, 0.01, size=(3, 3))
        # Make strain symmetric (physical strain is symmetric)
        strain = 0.5 * (strain + strain.T)
        # Apply strain to the cell: new_cell = (I + strain) @ old_cell
        old_cell = np.array(atoms.get_cell())
        new_cell = (np.eye(3) + strain) @ old_cell
        atoms.set_cell(new_cell, scale_atoms=False)

    # INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Structure {i:03d}: expected nat=2, got {nat}"
    assert ntyp == 2, f"Structure {i:03d}: expected ntyp=2, got {ntyp}"

    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)
    print(f"Wrote {out_path}: nat={nat}, ntyp={ntyp}")

print(f"\nDone. Generated {N_STRUCTURES} structures.")
