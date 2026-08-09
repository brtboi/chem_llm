import numpy as np
from pymatgen.core import Structure
from pathlib import Path

# SETTINGS
N_STRUCTURES = 51  # 50 perturbed + 1 base
TEMPERATURE = 300  # K
MAX_DISPLACEMENT = 0.2  # Å, upper bound for O atoms
TI_DISPLACEMENT = 0.1  # Å, upper bound for Ti atoms

# Random seed for reproducibility
np.random.seed(42)

# Base structure from Materials Project (rutile TiO2, space group P4_2/mnm)
BASE_CIF = "rutile_TiO2.cif"
STRUCTURE_DIR = "structures"

# Ensure output directory exists
Path(STRUCTURE_DIR).mkdir(exist_ok=True)

# Read the base structure
print(f"Reading base structure from {BASE_CIF}")
base_structure = Structure.from_file(BASE_CIF)

# Store the base structure (index 0)
base_structure.to(filename=Path(STRUCTURE_DIR) / "000.cif")
print(f"Saved base structure to {STRUCTURE_DIR}/000.cif")

# Generate 50 perturbed structures
for i in range(1, N_STRUCTURES):
    # Create a copy of the base structure
    perturbed_structure = base_structure.copy()

    # Get atomic positions in fractional coordinates
    frac_coords = perturbed_structure.frac_coords

    # Apply random displacements
    # For O atoms: random displacement up to MAX_DISPLACEMENT
    # For Ti atoms: random displacement up to TI_DISPLACEMENT
    # Use thermal displacement model: displacement ~ sqrt(kT / (m * omega^2))
    # Typical Debye-Waller factors for oxides suggest ~0.1–0.2 Å at 300 K
    displacements = []
    for j, site in enumerate(perturbed_structure.sites):
        if site.species_string == "Ti":
            # Ti displacement: smaller, more rigid
            displacement = np.random.uniform(-TI_DISPLACEMENT, TI_DISPLACEMENT, size=3)
        elif site.species_string == "O":
            # O displacement: larger, more mobile
            displacement = np.random.uniform(-MAX_DISPLACEMENT, MAX_DISPLACEMENT, size=3)
        else:
            # Skip other elements (shouldn't happen)
            displacement = np.zeros(3)
        displacements.append(displacement)

    # Convert displacements to fractional space and apply using translate_sites
    # This ensures proper handling of periodic boundary conditions
    for j, site in enumerate(perturbed_structure.sites):
        # Convert displacement from Cartesian to fractional
        # displacement_cart = displacement * lattice_vector
        # But we can directly use fractional displacement by dividing by lattice lengths
        # Since we're using fractional coordinates, we apply displacement directly in frac space
        frac_displacement = displacements[j] / perturbed_structure.lattice.abc
        perturbed_structure.translate_sites([j], frac_displacement, frac_coords=True, to_unit_cell=True)

    # Save the perturbed structure
    filename = Path(STRUCTURE_DIR) / f"{i:03d}.cif"
    perturbed_structure.to(filename=filename)
    print(f"Saved perturbed structure {i} to {filename}")

print("All structures generated.")
