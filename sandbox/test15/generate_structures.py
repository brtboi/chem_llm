import os
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter

# SETTINGS
N_STRUCTURES = 50

# Directory to save structures
os.makedirs("structures", exist_ok=True)

# Read the base rutile TiO2 structure from CIF
base_structure = Structure.from_file("rutile_TiO2.cif")

# Define room-temperature thermal displacement parameters (in Angstrom)
# Based on Debye-Waller factors for TiO2 at 300 K:
# Ti: ~0.08 Å, O: ~0.12 Å (typical for oxides)
displacement_map = {
    "Ti": 0.00,
    "O": 0.0
}

# Generate 50 perturbed structures
for n in range(N_STRUCTURES):
    # Create a copy of the base structure
    s = base_structure.copy()

    # Apply random displacements to each atom
    for i, site in enumerate(s):
        element = site.specie.symbol
        sigma = displacement_map[element]
        # Gaussian random displacement in Cartesian coordinates
        dr = np.random.normal(scale=sigma, size=3)
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

    # Save the perturbed structure as CIF
    filename = f"structures/structure_{n:03d}.cif"
    CifWriter(s).write_file(filename, mode='wt')
    print(f"Wrote {filename}")

print("All structures generated.")
