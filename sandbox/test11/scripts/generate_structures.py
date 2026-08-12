import os
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

# Configuration
BASE_CIF_PATH = 'structures/base_rutile.cif'
OUTPUT_DIR = 'structures/perturbed'
NUM_STRUCTURES = 51  # 50 perturbed + 1 base
DISPLACEMENT_STD = 0.07  # Standard deviation of atomic displacements in Å (room temperature thermal motion)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read the base structure from CIF
base_structure = Structure.from_file(BASE_CIF_PATH)

# Save the base structure (unperturbed)
base_cif_path = os.path.join(OUTPUT_DIR, '000_base.cif')
CifWriter(base_structure).write_file(base_cif_path)
print(f'Saved base structure: {base_cif_path}')

# Generate 50 perturbed structures
np.random.seed(42)  # For reproducibility
for i in range(1, NUM_STRUCTURES):
    # Create a copy of the base structure to avoid modifying the original
    perturbed_structure = base_structure.copy()

    # Generate random displacements for each atom
    # Use a normal distribution with mean 0 and standard deviation DISPLACEMENT_STD
    # Displacements are in fractional coordinates (unit cell basis)
    displacements = np.random.normal(0, DISPLACEMENT_STD, size=(len(perturbed_structure), 3))

    # Apply displacements to fractional coordinates
    # Note: We use the fact that Structure.frac_coords is a property that can be assigned to
    # after copying, as long as the structure is mutable (which it is after copy)
    new_frac_coords = perturbed_structure.frac_coords + displacements

    # Apply periodic boundary conditions: wrap coordinates into [0, 1)
    new_frac_coords = np.mod(new_frac_coords, 1.0)

    # Update the fractional coordinates
    perturbed_structure.frac_coords = new_frac_coords

    # Save the perturbed structure
    output_cif_path = os.path.join(OUTPUT_DIR, f'{i:03d}_perturbed.cif')
    CifWriter(perturbed_structure).write_file(output_cif_path)
    print(f'Saved perturbed structure: {output_cif_path}')

print(f'Generated {NUM_STRUCTURES} structures in {OUTPUT_DIR}')