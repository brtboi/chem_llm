"""
STEP 2: Generate 51 perturbed Al structures (structure_000.cif ... structure_050.cif).

structure_000.cif is the unperturbed base cell (MP's primitive cell for mp-134).
structure_001..050 are perturbed copies: random atomic displacements representative
of thermal motion at T = 300 K, plus a small random lattice strain.

Temperature choice: fcc Al (mp-134) is the stable phase of aluminium from 0 K up to
its melting point (933 K), so 300 K is well within its stability range. No high-
temperature phase is involved.

Displacement amplitudes (per species):
  Al: experimental Debye-Waller mean-square displacement U ~ 0.0035 A^2 at 300 K.
      Per-axis RMS displacement = sqrt(U) ~ 0.059 A.
      We use RMS = 0.06 A per axis (i.e. U = 0.0036 A^2).
      Since Al is the only species, there is only one amplitude.
      (For a multi-species system, lighter or more weakly bound atoms would get
       larger amplitudes; here all atoms are Al so the amplitude is uniform.)

Lattice strain:
  Small random symmetric strain tensor with components ~ 0.5% (0.005).
  Applied as: new_cell = (I + strain) @ old_cell

Units:
  - Displacements: Angstrom (ASE convention).
  - U (mean-square displacement): A^2.
  - RMS per-axis displacement: A.
  - Strain: dimensionless.
"""

import numpy as np
from ase.io import read, write
from ase import Atoms

# --- Parameters ---
T = 300.0  # K
# Per-species RMS displacement (A). Derived from Debye-Waller U ~ 0.0035 A^2 at 300 K.
# sqrt(0.0035) ~ 0.059 A; we round to 0.06 A.
RMS_DISPLACEMENT = {
    'Al': 0.06,  # A (per-axis RMS; U = 0.0036 A^2)
}
# Small random lattice strain magnitude (dimensionless, ~0.5%)
STRAIN_MAGNITUDE = 0.005

N_STRUCTURES = 51  # 000 (unperturbed) + 001..050 (perturbed)

# --- Read base structure ---
base = read('structures/Al_base.cif')
print(f"Base structure: {base.get_chemical_formula()}, nat={len(base)}, "
      f"ntyp={len(set(base.get_chemical_symbols()))}")

# INVARIANT 1: nat and ntyp computed from the actual Atoms object
nat_base = len(base)
types_base = set(base.get_chemical_symbols())
ntyp_base = len(types_base)
assert nat_base == 1, f"Expected 1 atom in base cell, got {nat_base}"
assert ntyp_base == 1, f"Expected 1 species in base cell, got {ntyp_base}"

rng = np.random.default_rng(seed=42)

for i in range(N_STRUCTURES):
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()
        
        # --- Random atomic displacements (thermal motion) ---
        # Per-axis RMS displacement for each species (A).
        # Displacements are drawn from N(0, RMS^2) independently per axis.
        displacements = np.zeros((len(atoms), 3))
        for j, sym in enumerate(atoms.get_chemical_symbols()):
            rms = RMS_DISPLACEMENT[sym]
            displacements[j] = rng.normal(0.0, rms, size=3)
        atoms.positions += displacements
        
        # --- Small random lattice strain ---
        # Symmetric strain tensor with components ~ STRAIN_MAGNITUDE.
        # strain = [[e11, e12, e13],
        #           [e12, e22, e23],
        #           [e13, e23, e33]]
        # Each component drawn from U(-STRAIN_MAGNITUDE, STRAIN_MAGNITUDE).
        e11 = rng.uniform(-STRAIN_MAGNITUDE, STRAIN_MAGNITUDE)
        e22 = rng.uniform(-STRAIN_MAGNITUDE, STRAIN_MAGNITUDE)
        e33 = rng.uniform(-STRAIN_MAGNITUDE, STRAIN_MAGNITUDE)
        e12 = rng.uniform(-STRAIN_MAGNITUDE, STRAIN_MAGNITUDE)
        e13 = rng.uniform(-STRAIN_MAGNITUDE, STRAIN_MAGNITUDE)
        e23 = rng.uniform(-STRAIN_MAGNITUDE, STRAIN_MAGNITUDE)
        strain = np.array([
            [e11, e12, e13],
            [e12, e22, e23],
            [e13, e23, e33]
        ])
        # Apply strain: new_cell = (I + strain) @ old_cell
        # In ASE, cell is a 3x3 matrix of column vectors (a, b, c).
        # (I + strain) @ cell transforms the cell vectors.
        new_cell = (np.eye(3) + strain) @ atoms.cell.array
        atoms.set_cell(new_cell, scale_atoms=True)
    
    # INVARIANT 1: assert nat/ntyp for each structure
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == nat_base, f"Structure {i:03d}: nat={nat}, expected {nat_base}"
    assert ntyp == ntyp_base, f"Structure {i:03d}: ntyp={ntyp}, expected {ntyp_base}"
    
    out_path = f'structures/structure_{i:03d}.cif'
    write(out_path, atoms)
    print(f"Wrote {out_path}: nat={nat}, ntyp={ntyp}")

print(f"\nDone: wrote {N_STRUCTURES} structures (000 unperturbed, 001-050 perturbed).")
print(f"Temperature: {T} K, RMS displacement Al: {RMS_DISPLACEMENT['Al']} A, "
      f"strain magnitude: {STRAIN_MAGNITUDE}")
