from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
import os

# Define output directory
output_dir = "interpolated_structures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read the two endpoint CIF files
print("Reading cubic CsPbBr3 structure (Pm-3m)...")
cubic_structure = Structure.from_file("cubic_cspbr3.cif")

print("Reading orthorhombic CsPbBr3 structure (Pnma)...")
orthorhombic_structure = Structure.from_file("orthorhombic_cspbr3.cif")

# Step 1: Align both structures into a common supercell
# The cubic structure has Z=1, orthorhombic has Z=4
# We will expand both to a 2x2x2 supercell to ensure both have 40 atoms (10 formula units)

print("Expanding cubic structure to 2x2x2 supercell...")
cubic_supercell = cubic_structure.make_supercell([2, 2, 2])
print(f"Cubic supercell: {len(cubic_supercell)} atoms")

print("Expanding orthorhombic structure to 2x2x2 supercell...")
orthorhombic_supercell = orthorhombic_structure.make_supercell([2, 2, 2])
print(f"Orthorhombic supercell: {len(orthorhombic_supercell)} atoms")

# Step 2: Use StructureMatcher to find the best atom correspondence
# We use a tolerance of 0.1 Å and allow for symmetry operations
print("Finding atom correspondence using StructureMatcher...")
sm = StructureMatcher(
    attempt_supercell=False,  # We already made supercells
    ltol=0.1,  # length tolerance
    stol=0.1,  # angle tolerance
    angle_tol=5,  # angle tolerance
    primitive_cell=False,
    scale=False
)

# Get the best match
match = sm.fit(cubic_supercell, orthorhombic_supercell)

if not match:
    raise ValueError("No valid match found between the two structures.")

# Extract the mapping of atoms from cubic to orthorhombic
mapping = match.mapping
print(f"Atom mapping established: {len(mapping)} atom pairs.")

# Step 3: Interpolate lattice and atomic positions
# Number of intermediate structures
n_interpolations = 50

# Create a list to store all structures
structures = []

# Interpolate from cubic (0) to orthorhombic (1)
for i in range(n_interpolations + 1):
    t = i / n_interpolations  # interpolation parameter from 0 to 1

    # Interpolate lattice vectors
    lattice = Lattice(
        (1 - t) * cubic_supercell.lattice.matrix +
        t * orthorhombic_supercell.lattice.matrix
    )

    # Interpolate atomic positions
    # Use the mapping to ensure consistent atom ordering
    frac_coords = []
    for j, (cubic_idx, ortho_idx) in enumerate(mapping):
        # Interpolate fractional coordinates
        frac_coord = (1 - t) * cubic_supercell.frac_coords[cubic_idx] + \
                     t * orthorhombic_supercell.frac_coords[ortho_idx]
        frac_coords.append(frac_coord)

    # Create new structure
    new_structure = Structure(
        lattice=lattice,
        species=cubic_supercell.species,
        coords=frac_coords,
        coords_are_cartesian=False
    )

    # Set the formula to ensure correct labeling
    new_structure = new_structure.get_primitive_structure()
    new_structure = new_structure.copy()
    new_structure.set_charge(0)

    # Save as CIF
    filename = os.path.join(output_dir, f"cspbr3_interpolated_{i:03d}.cif")
    new_structure.to(filename=filename, fmt="cif")
    structures.append(new_structure)

    print(f"Saved interpolated structure {i} ({t:.2f})")

print(f"Successfully generated {len(structures)} CIF files in {output_dir}/")

# Final verification: read one intermediate file and check
print("Verifying one intermediate structure...")
intermediate_file = os.path.join(output_dir, "cspbr3_interpolated_25.cif")
if os.path.exists(intermediate_file):
    with open(intermediate_file, 'r') as f:
        content = f.read()
    if "_cell_length_a" in content and "_atom_site_fract_x" in content:
        print("Intermediate CIF appears valid.")
    else:
        print("Warning: Intermediate CIF content seems malformed.")
else:
    print("Error: Intermediate CIF file not found.")

print("All done.")