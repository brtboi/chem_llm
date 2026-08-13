import os
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter

# SETTINGS
N_STRUCTURES = 50

# Directory to save structures
os.makedirs("structures", exist_ok=True)

# Base structure: rutile TiO2 from Materials Project (mp-2657)
# Space group: P4_2/mnm (136)
# Reference: https://materialsproject.org/materials/mp-2657

# Thermal displacement parameters at room temperature (estimated from literature):
# - Ti: ~0.03–0.06 Å (Gaussian sigma ~0.045 Å)
# - O: ~0.05–0.10 Å (Gaussian sigma ~0.075 Å)
# These values represent root-mean-square displacements due to thermal vibrations.

# We use a Gaussian distribution with these standard deviations for random atomic displacements.
displacement_sigma = {
    "Ti": 0.045,  # Å
    "O": 0.075   # Å
}

# Read the base structure from CIF
base_structure = Structure.from_file("structures/rutile_TiO2.cif")

# Ensure the base structure is in fractional coordinates
base_structure.make_supercell([1, 1, 1])  # Ensure primitive cell

# Generate 50 perturbed structures
for n in range(N_STRUCTURES):
    # Create a copy of the base structure
    s = base_structure.copy()

    # Apply random displacements to each atom
    for i, site in enumerate(s):
        # Get element symbol
        element = site.specie.symbol
        # Get displacement sigma for this element
        sigma = displacement_sigma[element]
        # Generate random displacement vector (in Cartesian coordinates)
        dr = np.random.normal(scale=sigma, size=3)
        # Apply displacement
        s.translate_sites(i, dr, frac_coords=False)

    # Optional: small random strain (to avoid perfect symmetry)
    eps = np.random.uniform(-0.005, 0.005, size=3)
    strain = np.diag(1 + eps)
    new_matrix = strain @ s.lattice.matrix
    s = Structure(
        lattice=Lattice(new_matrix),
        species=s.species,
        coords=s.cart_coords,
        coords_are_cartesian=True
    )

    # Save as CIF
    filename = f"structures/structure_{n:03d}.cif"
    CifWriter(s).write_file(filename, mode='wt')
    print(f"Wrote {filename}")

print("All structures generated.")
