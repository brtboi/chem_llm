import os
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter

# SETTINGS
N_STRUCTURES = 50

# Directory setup
os.makedirs("structures", exist_ok=True)

# Read the base structure from Materials Project (F-43m GaAs)
base_structure = Structure.from_file("structures/gaas_base.cif")

# Define displacement parameters in Angstroms
# Based on room temperature thermal vibrations (~0.1-0.2 Å) for Ga and As
# Typical Debye-Waller factors suggest atomic displacements ~0.15 Å at 300K
# We use a slightly higher value to ensure diversity while remaining chemically reasonable
DISPLACEMENT_MAP = {
    "Ga": 0.20,
    "As": 0.18
}

# Generate 50 perturbed structures
for n in range(N_STRUCTURES):
    # Create a copy of the base structure
    s = base_structure.copy()

    # Apply random displacements to each atom
    # Use Gaussian noise with standard deviation from DISPLACEMENT_MAP
    # Displacements are in Cartesian coordinates (Angstroms)
    for i, site in enumerate(s):
        atom_type = site.specie.symbol
        sigma = DISPLACEMENT_MAP[atom_type]
        dr = np.random.normal(scale=sigma, size=3)  # Random displacement in Cartesian space
        s.translate_sites(i, dr, frac_coords=False)

    # Optional: small random strain (not required but can add diversity)
    # eps = np.random.uniform(-0.01, 0.01, size=3)
    # strain = np.diag(1 + eps)
    # new_matrix = strain @ s.lattice.matrix
    # s = Structure(lattice=Lattice(new_matrix), species=s.species, coords=s.cart_coords, coords_are_cartesian=True)

    # Save the perturbed structure as a CIF file
    filename = f"structures/structure_{n:03d}.cif"
    CifWriter(s).write_file(filename, mode='wt')
    print(f"Wrote {filename}")

print("All structures generated.")
