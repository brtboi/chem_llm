import numpy as np
import ase.io

# =============================================================================
# generate_structures.py
# Generates 51 perturbed TiO2 structures (structure_000.cif to structure_050.cif)
# =============================================================================

# Temperature choice:
# Rutile TiO2 (alpha phase) is the thermodynamically stable phase at room temperature
# and remains stable up to ~1000 K. Beta-TiO2 (bronze structure) exists only at high T.
# We use T = 300 K, well within the rutile stability range.
T = 300.0  # K

# Einstein model for thermal displacements:
# U (crystallographic displacement parameter) = mean-square displacement in A^2
# Per-axis RMS displacement = sqrt(U) in A
#
# For TiO2 at 300 K, typical U values from neutron diffraction:
#   O: U ~ 0.005 A^2  -> RMS = sqrt(0.005) ~ 0.071 A
#   Ti: U ~ 0.002 A^2 -> RMS = sqrt(0.002) ~ 0.045 A
#
# Lighter O atoms move more than heavier Ti atoms (47.87 amu vs 16.00 amu).
# These values are representative of thermal motion at 300 K.

U_O = 0.005   # A^2, mean-square displacement for O
U_Ti = 0.002  # A^2, mean-square displacement for Ti

rms_O = np.sqrt(U_O)   # A, per-axis RMS displacement for O
rms_Ti = np.sqrt(U_Ti) # A, per-axis RMS displacement for Ti

# Small random lattice strain: 0.1% (typical for thermal expansion effects)
strain_magnitude = 0.001  # dimensionless (0.1%)

# Read the base structure (already reduced to primitive cell)
base = ase.io.read('structures/structure_000.cif')
print(f"Base structure: {len(base)} atoms, cell = {base.cell.array}")

# Verify base structure
assert len(base) == 6, f"Expected 6 atoms in primitive cell, got {len(base)}"
assert len(set(base.get_chemical_symbols())) == 2, "Expected Ti and O"

# Write structure_000 (unperturbed base)
ase.io.write('structures/structure_000.cif', base)
print("Wrote structure_000.cif (unperturbed)")

# Generate perturbed structures
np.random.seed(42)  # Reproducibility

for i in range(1, 51):
    atoms = base.copy()
    
    # Apply random atomic displacements (thermal motion)
    # Displacements are in Cartesian coordinates (A)
    displacements = np.zeros((len(atoms), 3))
    for j, sym in enumerate(atoms.get_chemical_symbols()):
        if sym == 'O':
            rms = rms_O
        elif sym == 'Ti':
            rms = rms_Ti
        else:
            raise ValueError(f"Unknown element: {sym}")
        displacements[j] = np.random.normal(0.0, rms, size=3)
    
    atoms.positions += displacements
    
    # Apply small random lattice strain
    # Strain tensor is symmetric, traceless (volume-preserving) to avoid
    # changing the density significantly
    strain = np.random.normal(0.0, strain_magnitude, size=(3, 3))
    strain = (strain + strain.T) / 2.0  # Symmetrize
    trace = np.trace(strain)
    strain -= (trace / 3.0) * np.eye(3)  # Make traceless
    
    # Apply strain to the cell: new_cell = old_cell @ (I + strain).T
    # This deforms the cell vectors while keeping fractional coordinates fixed
    new_cell = atoms.cell.array @ (np.eye(3) + strain).T
    atoms.set_cell(new_cell, scale_atoms=False)
    
    # Verify nat/ntyp INVARIANT per structure
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 6, f"Structure {i:03d}: expected nat=6, got {nat}"
    assert ntyp == 2, f"Structure {i:03d}: expected ntyp=2, got {ntyp}"
    
    # Write the structure
    fname = f'structures/structure_{i:03d}.cif'
    ase.io.write(fname, atoms)
    
    if i % 10 == 0:
        print(f"Wrote {fname}")

print("Done: wrote 51 structures (structure_000.cif to structure_050.cif)")
