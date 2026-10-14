import numpy as np
import ase.io
from ase import Atoms
from chem_llm.kpath import standard_kpath, kp_atoms

# =============================================================================
# generate_structures.py
# Generates 51 perturbed rutile TiO2 structures (structure_000 to structure_050)
# =============================================================================

# Temperature choice: T = 300 K
# Rutile TiO2 is the stable phase at room temperature and up to ~860 C (1133 K).
# At 300 K, thermal displacements are small and well-described by the harmonic
# approximation. This is a clear semiconductor (gap ~3 eV in experiment, ~1.77 eV
# in DFT-PBE), non-magnetic, so no special occupations or spin settings needed.

# =============================================================================
# Thermal displacement amplitudes (RMS, in Angstrom)
# =============================================================================
# For a harmonic oscillator at temperature T, the mean-square displacement is:
#   <u^2> = 3 * k_B * T / (M * omega^2)
# where omega is the characteristic vibrational frequency.
#
# We use a simplified Debye-model estimate:
#   u_rms = sqrt(3 * k_B * T / (M * omega_D^2))
# where omega_D is the Debye frequency.
#
# For TiO2 rutile, typical Debye frequencies are:
#   Ti: ~400 cm^-1  (heavier, stiffer bond -> smaller displacement)
#   O:  ~600 cm^-1  (lighter, more mobile -> larger displacement)
#
# Converting cm^-1 to s^-1: omega = 2*pi*c*nu, c = 2.998e10 cm/s
# k_B = 1.381e-23 J/K, M in kg
#
# For Ti (M = 47.87 g/mol = 7.95e-26 kg):
#   omega_D = 2*pi*2.998e10*400 = 7.55e13 s^-1
#   u_rms = sqrt(3 * 1.381e-23 * 300 / (7.95e-26 * (7.55e13)^2))
#         = sqrt(1.243e-20 / 4.53e-1) = sqrt(2.74e-20) = 1.66e-10 m = 0.166 A
#
# For O (M = 16.00 g/mol = 2.66e-26 kg):
#   omega_D = 2*pi*2.998e10*600 = 1.13e14 s^-1
#   u_rms = sqrt(3 * 1.381e-23 * 300 / (2.66e-26 * (1.13e14)^2))
#         = sqrt(1.243e-20 / 3.40e-1) = sqrt(3.66e-20) = 1.91e-10 m = 0.191 A
#
# These are RMS displacements per axis (i.e., sqrt(U) where U is the
# crystallographic mean-square displacement parameter in A^2).
# We use slightly rounded values:
T = 300  # K
U_Ti = 0.17**2  # A^2, mean-square displacement for Ti (RMS = 0.17 A)
U_O  = 0.19**2  # A^2, mean-square displacement for O  (RMS = 0.19 A)

# Lattice strain: small random strain, ~0.1% (typical of thermal expansion
# fluctuations at 300 K for a stiff oxide like rutile TiO2)
STRAIN_MAX = 0.001  # 0.1% maximum strain component

# Number of structures
N_STRUCTURES = 51  # 000 (unperturbed) + 50 perturbed

# Read the base structure (already reduced to primitive cell)
base = ase.io.read('structures/structure_000.cif')
print(f"Base structure: {len(base)} atoms, cell={base.cell}")

# Verify INVARIANT 1: nat and ntyp from the actual Atoms object
nat = len(base)
typ_set = set(base.get_chemical_symbols())
ntyp = len(typ_set)
print(f"nat={nat}, ntyp={ntyp}, species={typ_set}")
assert nat == 6, f"Expected 6 atoms in primitive rutile TiO2, got {nat}"
assert ntyp == 2, f"Expected 2 species (Ti, O), got {ntyp}"

# Set up random seed for reproducibility
rng = np.random.default_rng(seed=42)

for i in range(N_STRUCTURES):
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()
        
        # --- Random atomic displacements (thermal motion) ---
        # Displace each atom by a Gaussian random vector with per-species RMS
        # amplitude. The displacement is in Cartesian coordinates (Angstrom).
        # U is the mean-square displacement in A^2, so RMS = sqrt(U) in A.
        positions = atoms.get_positions().copy()
        for j, sym in enumerate(atoms.get_chemical_symbols()):
            if sym == 'Ti':
                rms = np.sqrt(U_Ti)  # A, RMS displacement per axis
            elif sym == 'O':
                rms = np.sqrt(U_O)   # A, RMS displacement per axis
            else:
                rms = 0.15  # fallback
            # Gaussian displacement: each Cartesian component has std = rms
            disp = rng.normal(0.0, rms, size=3)
            positions[j] += disp
        atoms.set_positions(positions)
        
        # --- Small random lattice strain ---
        # Apply a random strain tensor to the cell. The strain is small
        # (~0.1%) and isotropic in distribution (each component drawn from
        # a uniform distribution in [-STRAIN_MAX, STRAIN_MAX]).
        # This represents thermal expansion fluctuations.
        strain = rng.uniform(-STRAIN_MAX, STRAIN_MAX, size=(3, 3))
        # Make strain symmetric (physical strain tensor is symmetric)
        strain = 0.5 * (strain + strain.T)
        # New cell = (I + strain) @ old_cell
        old_cell = np.array(atoms.cell)
        new_cell = (np.eye(3) + strain) @ old_cell
        atoms.set_cell(new_cell, scale_atoms=True)
    
    # INVARIANT 1: verify nat/ntyp for each structure
    assert len(atoms) == nat, f"Structure {i:03d}: nat changed from {nat}"
    assert len(set(atoms.get_chemical_symbols())) == ntyp, \
        f"Structure {i:03d}: ntyp changed from {ntyp}"
    
    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)
    
    if i == 0:
        print(f"Wrote {out_path} (unperturbed)")
    else:
        print(f"Wrote {out_path} (perturbed)")

print(f"\nDone: wrote {N_STRUCTURES} structures (structure_000 to structure_{N_STRUCTURES-1:03d})")
