import os
from pathlib import Path
from pymatgen.core import Structure
import numpy as np

# SETTINGS
SEED = 42
NUM_STRUCTURES = 50
OUTPUT_DIR = "structures"

# Set random seed for reproducibility
np.random.seed(SEED)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read the base structure from CIF file
base_structure = Structure.from_file("rutile_TiO2.cif")

# Define the maximum displacement in Angstroms (room temperature thermal vibrations)
# Typical atomic displacements at room temperature are ~0.05 - 0.15 Å for oxides
MAX_DISPLACEMENT = 0.15  # in Angstroms

# Generate 50 perturbed structures
for i in range(NUM_STRUCTURES):
    # Create a copy of the base structure
    perturbed_structure = base_structure.copy()

    # Get fractional coordinates
    frac_coords = perturbed_structure.frac_coords

    # Generate random displacements for each atom
    # Use a normal distribution centered at 0 with standard deviation ~0.05 Å
    # This mimics room temperature thermal motion
    displacements = np.random.normal(loc=0.0, scale=0.05, size=frac_coords.shape)

    # Apply displacements
    new_frac_coords = frac_coords + displacements

    # Ensure coordinates remain within [0, 1) using modulo
    new_frac_coords = np.mod(new_frac_coords, 1.0)

    # Directly update the fractional coordinates of the structure
    perturbed_structure.frac_coords = new_frac_coords

    # Save the perturbed structure as a CIF file
    cif_filename = f"{OUTPUT_DIR}/TiO2_rutile_perturbed_{i:03d}.cif"
    perturbed_structure.to(filename=cif_filename, fmt="cif")

    # Optional: Print progress
    if i % 10 == 0:
        print(f"Generated {i} structures...")

print("All structures generated.")