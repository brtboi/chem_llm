import ase.io
import numpy as np
from chem_llm.kpath import standard_kpath, kp_atoms

# ============================================================
# generate_structures.py
# Generate 51 perturbed h-BN structures for band-structure study.
# structure_000.cif = unperturbed primitive cell
# structure_001..050.cif = random thermal displacements + small strain
# ============================================================

# --- Temperature choice ---
# h-BN is stable at room temperature and above (melting point ~3000 K).
# We use T = 300 K, a temperature at which h-BN is the thermodynamically
# stable phase (no phase transition occurs below ~1000 K).
T = 300.0  # Kelvin

# --- Debye-Waller amplitudes (RMS displacement per axis, in Angstrom) ---
# For a harmonic oscillator: <u^2> = 3kT / (M * omega^2)
# We use literature-typical U_iso values for h-BN at 300 K:
#   B: U_iso ~ 0.005 A^2  -> RMS = sqrt(0.005) ~ 0.071 A
#   N: U_iso ~ 0.004 A^2  -> RMS = sqrt(0.004) ~ 0.063 A
# These are mean-square displacements (U) in A^2; the per-axis RMS is sqrt(U) in A.
# B is lighter (10.8 u) than N (14.0 u), so B moves slightly more.
U_B = 0.005   # A^2, mean-square displacement for B
U_N = 0.004   # A^2, mean-square displacement for N
rms_B = np.sqrt(U_B)  # A, per-axis RMS displacement for B
rms_N = np.sqrt(U_N)  # A, per-axis RMS displacement for N

# --- Lattice strain ---
# Small random strain to represent thermal expansion / cell distortion.
# Typical thermal strain in h-BN at 300 K is ~1e-4 to 1e-3.
# We use a uniform random strain in [-0.001, 0.001] (0.1% max).
strain_max = 0.001  # dimensionless, 0.1%

# --- Seed for reproducibility ---
np.random.seed(42)

# --- Read base structure and reduce to primitive cell ---
base = ase.io.read('structures/BN_base.cif')
kp = standard_kpath(base)
prim = kp_atoms(kp)  # primitive cell (4 atoms for h-BN)

# INVARIANT 1: nat/ntyp from actual object
nat = len(prim)
ntyp = len(set(prim.get_chemical_symbols()))
assert nat == 4, f'Expected 4 atoms, got {nat}'
assert ntyp == 2, f'Expected 2 species, got {ntyp}'

# --- Write structure_000.cif (unperturbed) ---
ase.io.write('structures/structure_000.cif', prim)
print(f'Wrote structures/structure_000.cif (unperturbed, nat={nat})')

# --- Generate perturbed structures ---
for i in range(1, 51):
    atoms = prim.copy()
    
    # Random atomic displacements (thermal motion)
    # Each atom gets a Gaussian displacement with species-specific RMS
    symbols = atoms.get_chemical_symbols()
    displacements = np.zeros((nat, 3))
    for j, sym in enumerate(symbols):
        if sym == 'B':
            rms = rms_B
        elif sym == 'N':
            rms = rms_N
        else:
            raise ValueError(f'Unexpected species {sym}')
        displacements[j] = np.random.normal(0.0, rms, size=3)
    
    # Apply displacements in Cartesian coordinates
    atoms.positions += displacements
    
    # Small random lattice strain
    # Apply a random symmetric strain tensor to the cell
    strain = np.random.uniform(-strain_max, strain_max, size=(3, 3))
    # Make strain symmetric (physical strain tensor is symmetric)
    strain = 0.5 * (strain + strain.T)
    # Apply strain to cell vectors: new_cell = (I + strain) @ old_cell
    old_cell = atoms.get_cell()
    new_cell = (np.eye(3) + strain) @ old_cell
    atoms.set_cell(new_cell, scale_atoms=True)
    
    # INVARIANT 1: verify nat/ntyp for each structure
    assert len(atoms) == nat, f'Structure {i:03d}: nat changed'
    assert len(set(atoms.get_chemical_symbols())) == ntyp, f'Structure {i:03d}: ntyp changed'
    
    fname = f'structures/structure_{i:03d}.cif'
    ase.io.write(fname, atoms)
    print(f'Wrote {fname} (nat={len(atoms)}, ntyp={ntyp})')

print('Done: 51 structures written (structure_000.cif through structure_050.cif)')
