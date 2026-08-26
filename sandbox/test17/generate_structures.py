import os
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter

# SETTINGS
N_STRUCTURES = 50

# Output directory
os.makedirs("structures", exist_ok=True)

# Base structure: rutile TiO2
base_structure = Structure.from_file("structures/rutile_TiO2.cif")

# Displacement parameters (in angstrom) for room temperature thermal motion
# Typical values for TiO2: Ti ~0.15 Å, O ~0.20 Å
DISPLACEMENT_MAP = {
    "Ti": 0.15,
    "O": 0.20
}

# Generate 50 perturbed structures
for n in range(N_STRUCTURES):
    # Create a copy of the base structure
    s = base_structure.copy()

    # Apply random atomic displacements
    for i, site in enumerate(s):
        element = site.specie.symbol
        sigma = DISPLACEMENT_MAP[element]
        # Draw displacement from normal distribution
        dr = np.random.normal(scale=sigma, size=3)
        # Apply displacement in Cartesian coordinates
        s.translate_sites(i, dr, frac_coords=False)

    # Optional: small random strain (to avoid perfect symmetry)
    eps = np.random.uniform(-0.01, 0.01, size=3)
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
    CifWriter(s).write_file(filename)
    print(f"Wrote {filename}")

print("All structures generated.")
