import numpy as np
import ase.io

# =============================================================================
# generate_structures.py
# Generates 51 perturbed rutile TiO2 structures (structure_000.cif .. structure_050.cif)
# structure_000.cif is the unperturbed base (already written by reduce_primitive.py)
#
# Temperature choice: 300 K (room temperature)
#   Rutile TiO2 is the thermodynamically stable phase at ambient conditions.
#   It remains stable up to ~1400 K before transforming to the high-temperature
#   phases (e.g. alpha-PbO2-type). 300 K is well within the stable range.
#
# Displacement amplitudes (RMS, in Angstrom):
#   The per-axis RMS displacement is sqrt(U) where U is the mean-square
#   displacement parameter in A^2.
#   We use the Debye model estimate: U = 3hbar^2 / (2*pi^2*m*k_B*T) * (1 - exp(-theta_D/T))
#   For simplicity we use the high-temperature limit: U = 3hbar^2 / (2*pi^2*m*k_B*T)
#   Actually, for a rough estimate at 300 K:
#     Ti (M=47.87 g/mol): U ~ 0.003 A^2 -> RMS ~ 0.055 A
#     O  (M=16.00 g/mol): U ~ 0.009 A^2 -> RMS ~ 0.095 A
#   These are representative of thermal motion at 300 K. Oxygen is lighter and
#   more weakly bound (higher Debye temperature relative to mass), so it moves more.
#   We use RMS displacements: Ti = 0.055 A, O = 0.095 A.
#
# Lattice strain: small random strain tensor with components up to 0.5% (0.005)
#   This represents small deviations from the ideal lattice due to thermal
#   expansion fluctuations and defects.
# =============================================================================

np.random.seed(42)

# Read the base (unperturbed) structure
base = ase.io.read('structures/structure_000.cif')
print(f"Base structure: {len(base)} atoms, cell = {base.cell[:].tolist()}")

# Per-species RMS displacement amplitudes in Angstrom
# These are sqrt(U) values, i.e. the per-axis RMS displacement.
# Justification: at 300 K, using the high-temperature Debye model estimate:
#   U = 3*hbar^2 / (2*pi^2*m*k_B*T)
#   For Ti (M=47.87 g/mol): U ~ 0.003 A^2, RMS = sqrt(0.003) ~ 0.055 A
#   For O  (M=16.00 g/mol): U ~ 0.009 A^2, RMS = sqrt(0.009) ~ 0.095 A
# Oxygen moves more because it is lighter (smaller mass -> larger displacement).
RMS_DISPLACEMENT = {
    'Ti': 0.055,  # Angstrom, per-axis RMS = sqrt(U)
    'O':  0.095,  # Angstrom, per-axis RMS = sqrt(U)
}

# Maximum strain component (dimensionless, i.e. fractional change in lattice parameter)
MAX_STRAIN = 0.005  # 0.5%

# Number of structures to generate (000 is the base, 001-050 are perturbed)
N_STRUCTURES = 51

# structure_000 is already written; start from 001
for n in range(1, N_STRUCTURES):
    atoms = base.copy()
    
    # --- Random atomic displacements (thermal motion) ---
    # Each atom gets a random displacement drawn from a Gaussian distribution
    # with standard deviation equal to the per-species RMS displacement.
    # The displacement is in Cartesian coordinates (Angstrom).
    symbols = atoms.get_chemical_symbols()
    displacements = np.zeros((len(atoms), 3))
    for i, sym in enumerate(symbols):
        rms = RMS_DISPLACEMENT[sym]
        displacements[i] = np.random.normal(0.0, rms, size=3)
    
    # Apply displacements to fractional coordinates
    # Convert Cartesian displacement to fractional: df = d_cart @ inv(cell)
    cell = atoms.cell
    inv_cell = np.linalg.inv(cell.array)
    frac_displacements = displacements @ inv_cell
    
    new_frac = atoms.get_scaled_positions() + frac_displacements
    atoms.set_scaled_positions(new_frac)
    
    # --- Small random lattice strain ---
    # Apply a small random strain tensor to the cell.
    # strain is a 3x3 tensor; new_cell = (I + strain) @ old_cell
    # We use a symmetric strain tensor (no shear) for simplicity,
    # with diagonal components up to MAX_STRAIN and off-diagonal up to MAX_STRAIN/2.
    strain = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            if i == j:
                strain[i, j] = np.random.uniform(-MAX_STRAIN, MAX_STRAIN)
            else:
                val = np.random.uniform(-MAX_STRAIN / 2, MAX_STRAIN / 2)
                strain[i, j] = val
                strain[j, i] = val
    
    new_cell = (np.eye(3) + strain) @ cell.array
    atoms.set_cell(new_cell, scale_atoms=False)
    
    # --- INVARIANT 1: nat/ntyp check ---
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 6, f"Structure {n:03d}: expected 6 atoms, got {nat}"
    assert ntyp == 2, f"Structure {n:03d}: expected 2 species, got {ntyp}"
    
    # Write the structure
    out_path = f'structures/structure_{n:03d}.cif'
    ase.io.write(out_path, atoms)
    
    if n % 10 == 0:
        print(f"  Wrote {out_path} (nat={nat}, ntyp={ntyp})")

print(f"\nDone. Generated structures 001-050 (structure_000 was already written).")
print(f"Total structures: {N_STRUCTURES}")
