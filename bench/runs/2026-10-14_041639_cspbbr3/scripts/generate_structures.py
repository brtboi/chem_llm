import numpy as np
import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

# =============================================================================
# generate_structures.py
# Generates 51 perturbed CsPbBr3 structures (structure_000.cif - structure_050.cif)
# structure_000 is the unperturbed primitive cell; 001-050 have random thermal
# displacements + small random lattice strain.
# =============================================================================

# --- Temperature choice ---
# CsPbBr3 cubic (Pm-3m) phase is the room-temperature phase.
# The tetragonal phase appears below ~230 K; orthorhombic below ~160 K.
# T = 300 K is therefore appropriate for the cubic phase.
T = 300  # K

# --- Debye-Waller amplitudes at 300 K ---
# U is the mean-square displacement in A^2 (crystallographic displacement parameter).
# Per-axis RMS displacement = sqrt(U) in A.
# Heavier / more strongly bound atoms move less.
#   Cs:  ~133 amu, heaviest, U = 0.005 A^2 -> RMS = sqrt(0.005) ~ 0.071 A
#   Pb:  ~207 amu, heaviest, U = 0.003 A^2 -> RMS = sqrt(0.003) ~ 0.055 A
#   Br:  ~80 amu, lighter and more weakly bound in the halide framework,
#        U = 0.008 A^2 -> RMS = sqrt(0.008) ~ 0.089 A
U = {'Cs': 0.005, 'Pb': 0.003, 'Br': 0.008}  # A^2
rms = {sp: np.sqrt(u) for sp, u in U.items()}  # A (per-axis RMS)

# --- Lattice strain ---
# Small random symmetric strain tensor, magnitude ~1% (0.01).
# Applied as a linear transformation to the cell vectors.
STRAIN_MAG = 0.01  # dimensionless

# --- Seed for reproducibility ---
np.random.seed(42)

# --- Read base structure (already reduced to primitive cell) ---
base = ase.io.read('structures/structure_000.cif')
n_base = len(base)
ntyp_base = len(set(base.get_chemical_symbols()))
print(f'Base structure: {n_base} atoms, {ntyp_base} species')

# --- Generate 51 structures ---
for i in range(51):
    nnn = f'{i:03d}'
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()

        # --- Random atomic displacements (thermal motion) ---
        # Displace each atom along each Cartesian axis by a Gaussian random
        # variable with standard deviation = per-axis RMS for that species.
        # Displacements are in Cartesian coordinates (A).
        new_pos = np.array(atoms.positions, dtype=float)
        for idx, sp in enumerate(atoms.get_chemical_symbols()):
            new_pos[idx] += np.random.normal(0.0, rms[sp], size=3)
        atoms.positions = new_pos

        # --- Small random lattice strain ---
        # Generate a random symmetric 3x3 strain tensor with magnitude ~1%.
        # The strain tensor is dimensionless (delta_L / L).
        # Apply as: new_cell = old_cell @ (I + strain)^T
        # (cell vectors are rows of the cell matrix in ASE convention).
        #
        # Random symmetric tensor: A + A^T where A is random.
        A = np.random.randn(3, 3) * STRAIN_MAG / np.sqrt(6)
        strain = A + A.T  # symmetric, each element ~ N(0, STRAIN_MAG^2/3)
        # Rescale so the Frobenius norm equals STRAIN_MAG
        frob = np.linalg.norm(strain)
        if frob > 0:
            strain *= STRAIN_MAG / frob
        # Apply strain to cell vectors (rows of cell matrix)
        # new_cell_rows = old_cell_rows @ (I + strain)^T
        I = np.eye(3)
        atoms.cell = atoms.cell @ (I + strain).T
        atoms.pbc = True

    # --- INVARIANT 1: nat and ntyp must match the actual ASE Atoms object ---
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == n_base, f'structure_{nnn}: nat {nat} != base {n_base}'
    assert ntyp == ntyp_base, f'structure_{nnn}: ntyp {ntyp} != base {ntyp_base}'

    out_path = f'structures/structure_{nnn}.cif'
    ase.io.write(out_path, atoms)
    print(f'Wrote {out_path}: {nat} atoms, {ntyp} species')

print('Done: 51 structures written.')
