import os
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter

# SETTINGS
N_STRUCTURES = 50

# Output directory
os.makedirs("structures", exist_ok=True)

# Base structure from Materials Project (mp-66, diamond, Fd-3m space group)
# Read the CIF file to get the base structure
base_structure = Structure.from_file("diamond.cif")

# Define Debye-Waller-like thermal displacement parameters (in Angstroms) at room temperature
# For diamond (C atoms), based on experimental and DFT studies:
# - Carbon: ~0.03–0.06 Å at room temperature
# We use a conservative average: 0.05 Å
DISPLACEMENT_MAP = {
    "C": 0.05
}

# Function to generate a perturbed structure from the base structure
def perturb_structure(structure, seed=None):
    """
    Apply chemically reasonable random atomic displacements to a structure.
    
    Parameters:
    - structure: pymatgen Structure object
    - seed: optional random seed for reproducibility
    
    Returns:
    - perturbed Structure object
    """
    if seed is not None:
        np.random.seed(seed)

    # Create a copy to avoid modifying the original
    s = structure.copy()

    # Apply random displacements to each atom
    # Displacements are drawn from a normal distribution with standard deviation
    # based on the element-specific thermal parameters
    for i, site in enumerate(s):
        element = site.specie.symbol
        sigma = DISPLACEMENT_MAP[element]
        # Generate random displacement vector in Cartesian coordinates
        dr = np.random.normal(scale=sigma, size=3)
        # Apply displacement
        s.translate_sites(i, dr, frac_coords=False)

    return s

# Generate 50 perturbed structures (51 total including base)
print("Generating 51 diamond structures (1 base + 50 perturbed)...")

# Write the base structure first
base_filename = "structures/structure_000.cif"
CifWriter(base_structure).write_file(base_filename, mode='wt')
print(f"Wrote base structure: {base_filename}")

# Generate and write perturbed structures
for n in range(1, N_STRUCTURES + 1):
    # Use a different seed for each perturbation to ensure diversity
    seed = 42 + n  # Fixed seed for reproducibility
    perturbed_structure = perturb_structure(base_structure, seed=seed)
    filename = f"structures/structure_{n:03d}.cif"
    CifWriter(perturbed_structure).write_file(filename, mode='wt')
    print(f"Wrote perturbed structure: {filename}")

print("All structures generated successfully.")
