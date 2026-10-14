import numpy as np
import ase.io
import os

# =============================================================================
# generate_structures.py
# Generates 51 wurtzite GaN structures: structure_000.cif (unperturbed) and
# structure_001.cif .. structure_050.cif (perturbed with thermal displacements
# and small random lattice strain).
#
# Temperature: 300 K
#   Wurtzite GaN is the thermodynamically stable phase at room temperature.
#   It transforms to zincblende only at very high temperatures (>1000 K) or
#   under specific pressure conditions. 300 K is therefore a physically
#   meaningful choice for thermal-motion amplitudes.
#
# Per-species RMS displacement (per axis) in Angstrom:
#   These are sqrt(U) values, where U is the mean-square displacement in A^2.
#   Justification: typical Debye-Waller displacements for III-nitrides at 300 K.
#   Ga (M=69.723 g/mol): heavier, weaker effective spring -> smaller displacement
#   N  (M=14.007 g/mol): lighter, stronger effective spring -> larger displacement
#   RMS_Ga = 0.08 A  ->  U_Ga = 0.0064 A^2
#   RMS_N  = 0.12 A  ->  U_N  = 0.0144 A^2
#
# Strain: small random strain tensor with components ~0.5% (0.005).
#   This represents a small deviation from the ideal lattice, as would
#   be found in a real crystal with defects, impurities, or thermal
#   fluctuations.
# =============================================================================

np.random.seed(42)

# Read the base (unperturbed) structure
base = ase.io.read('structures/structure_000_base.cif')
print(f"Base structure: {len(base)} atoms, cell={base.cell[:].tolist()}")

# Verify nat/ntyp invariant for the base structure
nat_base = len(base)
types_base = set(base.get_chemical_symbols())
ntyp_base = len(types_base)
print(f"Base: nat={nat_base}, ntyp={ntyp_base}, types={types_base}")
assert nat_base == 4, f"Expected 4 atoms in base GaN, got {nat_base}"
assert ntyp_base == 2, f"Expected 2 types in base GaN, got {ntyp_base}"

# Per-species RMS displacement (per axis) in Angstrom
RMS_Ga = 0.08  # A (per-axis RMS displacement = sqrt(U_Ga))
RMS_N  = 0.12  # A (per-axis RMS displacement = sqrt(U_N))

# Mean-square displacements in A^2
U_Ga = RMS_Ga ** 2  # 0.0064 A^2
U_N  = RMS_N ** 2   # 0.0144 A^2

print(f"U_Ga = {U_Ga:.6f} A^2, rms_Ga = {RMS_Ga:.4f} A")
print(f"U_N  = {U_N:.6f} A^2, rms_N  = {RMS_N:.4f} A")

# Strain magnitude: 0.5% random strain
STRAIN_MAG = 0.005  # 0.5% strain

# Ensure output directory exists
os.makedirs('structures', exist_ok=True)

# Write structure_000.cif (unperturbed base)
ase.io.write('structures/structure_000.cif', base)
print("Wrote structures/structure_000.cif (unperturbed)")

# Generate 50 perturbed structures
for i in range(1, 51):
    atoms = base.copy()
    
    # Apply random lattice strain
    # Generate a small random symmetric strain tensor
    strain = np.random.normal(0, STRAIN_MAG, (3, 3))
    strain = (strain + strain.T) / 2  # make symmetric
    
    # Apply strain to the cell
    # new_cell = (I + strain) @ old_cell
    new_cell = (np.eye(3) + strain) @ atoms.cell[:]
    atoms.set_cell(new_cell, scale_atoms=True)
    
    # Apply random atomic displacements (thermal motion)
    # Each atom gets a random displacement drawn from a Gaussian distribution
    # with per-species standard deviation equal to the RMS displacement.
    positions = atoms.get_positions().copy()
    symbols = atoms.get_chemical_symbols()
    for j, sym in enumerate(symbols):
        if sym == 'Ga':
            disp = np.random.normal(0, RMS_Ga, 3)
        elif sym == 'N':
            disp = np.random.normal(0, RMS_N, 3)
        else:
            raise ValueError(f"Unknown element: {sym}")
        positions[j] += disp
    atoms.set_positions(positions)
    
    # Verify nat/ntyp invariant for this perturbed structure
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 4, f"Structure {i:03d}: expected 4 atoms, got {nat}"
    assert ntyp == 2, f"Structure {i:03d}: expected 2 types, got {ntyp}"
    
    # Write the perturbed structure
    fname = f'structures/structure_{i:03d}.cif'
    ase.io.write(fname, atoms)
    
    if i % 10 == 0:
        print(f"Wrote {fname}")

print("Done: wrote 51 structures (structure_000.cif through structure_050.cif)")
