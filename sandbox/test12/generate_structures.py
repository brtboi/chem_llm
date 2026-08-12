import os
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter

# SETTINGS
N_STRUCTURES = 50

# Directory to save structures
STRUCTURE_DIR = "structures"

# Ensure structure directory exists
os.makedirs(STRUCTURE_DIR, exist_ok=True)

# Base structure: rutile TiO2 from Materials Project (mp-2657)
# Space group: P4_2/mnm (136)
# Reference: https://materialsproject.org/materials/mp-2657

# Thermal displacement parameters at room temperature (estimated from literature):
# - Ti: ~0.03–0.06 Å (Gaussian sigma ~0.045 Å)
# - O: ~0.05–0.10 Å (Gaussian sigma ~0.075 Å)
# We use a slightly conservative sigma to ensure chemically reasonable perturbations.
DISPLACEMENT_SIGMA = {
    "Ti": 0.05,  # Å
    "O": 0.08   # Å
}

# Read base structure from CIF
base_structure = Structure.from_file("structures/base_rutile.cif")

# Function to apply random atomic displacements based on thermal vibrations
def perturb_structure(structure, sigma_map):
    """
    Apply random displacements to atoms using a Gaussian distribution.
    Displacements are applied in Cartesian coordinates.
    
    Args:
        structure: pymatgen Structure object
        sigma_map: dict mapping element symbol to standard deviation (Å)

    Returns:
        Perturbed Structure object
    """
    s = structure.copy()

    # Convert to Cartesian coordinates for displacement
    cart_coords = s.cart_coords

    # Apply random displacements
    for i, site in enumerate(s):
        element = site.specie.symbol
        sigma = sigma_map[element]
        # Generate random displacement vector (Gaussian, zero mean)
        dr = np.random.normal(scale=sigma, size=3)
        # Apply displacement in Cartesian space
        cart_coords[i] += dr

    # Update structure with new coordinates
    s = Structure(
        lattice=s.lattice,
        species=s.species,
        coords=cart_coords,
        coords_are_cartesian=True
    )

    return s

# Generate 50 perturbed structures
for n in range(N_STRUCTURES):
    # Create a copy of the base structure
    s = base_structure.copy()

    # Apply perturbation
    s = perturb_structure(s, DISPLACEMENT_SIGMA)

    # Save as CIF
    filename = os.path.join(STRUCTURE_DIR, f"structure_{n:03d}.cif")
    CifWriter(s).write_file(filename, mode='wt')
    print(f"Wrote {filename}")

print("All structures generated.")
