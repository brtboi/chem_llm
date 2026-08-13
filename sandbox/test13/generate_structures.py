import os
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter

# SETTINGS
N_STRUCTURES = 50

# Random seed for reproducibility
np.random.seed(42)

# Output directory
os.makedirs("structures", exist_ok=True)

# Base structure from Materials Project (mp-22862, NaCl, Fm-3m)
base_structure = Structure.from_file("base_structure.cif")

# Define reasonable RMS displacements for Na and Cl at room temperature (~0.05-0.1 Å)
# Based on thermal motion in ionic crystals; typical Debye-Waller factors
# We use 0.07 Å for Na and 0.08 Å for Cl as representative values
DISPLACEMENT_RMS = {
    "Na": 0.07,
    "Cl": 0.08
}

# Generate 50 perturbed structures
for n in range(N_STRUCTURES):
    # Copy the base structure to avoid modifying the original
    s = base_structure.copy()

    # Apply random displacements to each atom
    # Use normal distribution with per-atom standard deviation
    for i, site in enumerate(s):
        atom_type = site.specie.symbol
        sigma = DISPLACEMENT_RMS[atom_type]
        # Generate random displacement in Cartesian coordinates
        dr = np.random.normal(scale=sigma, size=3)
        s.translate_sites(i, dr, frac_coords=False)

    # Optional: small random strain to avoid perfect symmetry
    # This helps break degeneracies in electronic structure
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
