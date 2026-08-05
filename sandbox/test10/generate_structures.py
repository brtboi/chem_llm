import os
from pathlib import Path
from pymatgen.core import Structure, Site
import numpy as np

# SETTINGS
SEED = 42
N_STRUCTURES = 50
TEMPERATURE = 300  # K

# Output directory
STRUCTURE_DIR = "structures"
os.makedirs(STRUCTURE_DIR, exist_ok=True)

# Set random seed
np.random.seed(SEED)

# Read base structure from CIF file
base_structure = Structure.from_file("base_rutile_TiO2.cif")

# Generate 50 perturbed structures
for i in range(N_STRUCTURES):
    # Copy the base structure
    perturbed_structure = base_structure.copy()
    
    # Get fractional coordinates
    frac_coords = perturbed_structure.frac_coords
    
    # Generate random displacements based on room temperature
    # Use a standard deviation of 0.005 in fractional coordinates
    # This is a reasonable estimate for room temperature thermal motion
    # in rutile TiO2 (typical atomic displacements ~0.01-0.03 Å)
    displacements = np.random.normal(0, 0.005, size=frac_coords.shape)
    
    # Apply displacements
    new_frac_coords = frac_coords + displacements
    
    # Wrap coordinates back into [0,1)
    new_frac_coords = new_frac_coords % 1.0
    
    # Create new structure with updated fractional coordinates
    # This step ensures the new structure maintains the same space group
    # and avoids invalid fractional coordinates
    new_sites = [
        Site(site.species, new_coord)
        for site, new_coord in zip(perturbed_structure.sites, new_frac_coords)
    ]
    perturbed_structure = Structure(perturbed_structure.lattice, new_sites)
    
    # Save the perturbed structure
    cif_path = Path(STRUCTURE_DIR) / f"{i:03d}.cif"
    perturbed_structure.to(filename=str(cif_path))
    
    print(f"Saved perturbed structure {i:03d}")

print("All structures generated.")