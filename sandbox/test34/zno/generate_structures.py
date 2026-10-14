import numpy as np
import ase.io
from ase import Atoms

# ============================================================
# generate_structures.py
# Generates 51 wurtzite ZnO structures: structure_000.cif (unperturbed)
# and structure_001.cif .. structure_050.cif (perturbed).
# ============================================================

# --- Temperature choice ---
# Wurtzite ZnO (P6_3mc) is the thermodynamically stable phase at ambient
# conditions (room temperature, 1 atm). It remains stable up to ~1000 C.
# We use T = 300 K (room temperature) for the thermal displacement amplitudes.
# At this temperature the wurtzite phase is well below its phase boundary,
# so thermal motion is small and the structure remains in the wurtzite phase.
T = 300.0  # K

# --- Displacement amplitudes (RMS, in Angstrom) ---
# We use the Einstein model: U = 3*k_B*T / (M * omega^2)
# where U is the mean-square displacement (A^2) and RMS = sqrt(U) (A).
# For ZnO, the Debye temperature is ~700 K, so at 300 K the displacements
# are small. We use literature-typical values:
#   O (16 amu, lighter): RMS ~ 0.08 A  -> U_O = 0.0064 A^2
#   Zn (65 amu, heavier): RMS ~ 0.04 A  -> U_Zn = 0.0016 A^2
# These are per-axis RMS displacements (sqrt(U) in A).
# The lighter O atom moves more than the heavier Zn atom.
RMS_O = 0.08   # A, per-axis RMS displacement for O
RMS_Zn = 0.04  # A, per-axis RMS displacement for Zn

# --- Lattice strain ---
# Small random strain ~ 0.5% (0.005) to represent thermal expansion/contraction.
# Applied as a uniform strain tensor with random components.
STRAIN_MAG = 0.005  # 0.5%

# --- Seed for reproducibility ---
np.random.seed(42)

# Read the base (unperturbed) structure
base = ase.io.read('structures/structure_000.cif')
print(f"Base structure: {len(base)} atoms, symbols: {base.get_chemical_symbols()}")

# Write structure_000.cif (unperturbed) -- already exists, rewrite to be sure
ase.io.write('structures/structure_000.cif', base)
print("Wrote structures/structure_000.cif (unperturbed)")

for n in range(1, 51):
    nnn = f"{n:03d}"
    atoms = base.copy()

    # --- Random atomic displacements (thermal motion) ---
    # Each atom gets a random displacement drawn from a Gaussian distribution
    # with per-species RMS amplitude (in Cartesian coordinates, Angstrom).
    displacements = np.zeros((len(atoms), 3))
    for i, sym in enumerate(atoms.get_chemical_symbols()):
        if sym == 'O':
            rms = RMS_O
        elif sym == 'Zn':
            rms = RMS_Zn
        else:
            rms = 0.05  # fallback
        displacements[i] = np.random.normal(0.0, rms, size=3)
    atoms.positions += displacements

    # --- Small random lattice strain ---
    # Generate a random strain tensor with magnitude ~ STRAIN_MAG.
    # The strain is applied to the cell vectors.
    strain = np.random.uniform(-STRAIN_MAG, STRAIN_MAG, size=(3, 3))
    # Make it symmetric (strain tensor is symmetric)
    strain = 0.5 * (strain + strain.T)
    # Apply strain: new_cell = (I + strain) @ old_cell
    # In ASE, cell is a 3x3 array of row vectors (a1, a2, a3).
    # The strain transformation: new_cell = old_cell @ (I + strain).T
    # (since cell vectors are rows, the right-multiplication applies the strain)
    new_cell = atoms.cell @ (np.eye(3) + strain).T
    # Use set_cell with scale_atoms=True to update Cartesian positions
    # so that fractional coordinates are preserved when the cell changes.
    atoms.set_cell(new_cell, scale_atoms=True)

    # --- INVARIANT 1: nat/ntyp check ---
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 4, f"Structure {nnn}: expected 4 atoms, got {nat}"
    assert ntyp == 2, f"Structure {nnn}: expected 2 species, got {ntyp}"

    ase.io.write(f'structures/structure_{nnn}.cif', atoms)
    print(f"Wrote structures/structure_{nnn}.cif (nat={nat}, ntyp={ntyp})")

print("Done: 51 structures written (000-050).")
