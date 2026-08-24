import os
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter

# SETTINGS
N_PERTURBATIONS = 50

# Output directory
os.makedirs("structures", exist_ok=True)

# Read the base rutile TiO2 structure
base_structure = Structure.from_file("structures/rutile_TiO2.cif")

# Displacement magnitudes (in Angstrom) for Ti and O at room temperature
# Based on typical atomic vibrations: O ~ 0.05-0.1 Å, Ti ~ 0.03-0.07 Å
# Using 0.08 Å for O and 0.05 Å for Ti as reasonable estimates
DISPLACEMENT_MAP = {
    "Ti": 0.05,
    "O": 0.08
}

# Generate 50 perturbed structures (total 51 including base)
for i in range(N_PERTURBATIONS + 1):
    # Start with the base structure
    if i == 0:
        structure = base_structure.copy()
        filename = "structures/rutile_TiO2_base.cif"
    else:
        # Apply random displacements using pymatgen's randomize_structure
        # This function applies Gaussian displacements and a small random strain
        structure = base_structure.copy()
        for site in structure:
            sigma = DISPLACEMENT_MAP[site.specie.symbol]
            dr = np.random.normal(scale=sigma, size=3)
            structure.translate_sites(structure.sites.index(site), dr, frac_coords=False)

        # Small random strain (as in original script)
        eps = np.random.uniform(-0.015, 0.015, size=3)
        strain = np.diag(1 + eps)
        new_matrix = strain @ structure.lattice.matrix
        structure = Structure(
            lattice=Lattice(new_matrix),
            species=structure.species,
            coords=structure.cart_coords,
            coords_are_cartesian=True
        )

        filename = f"structures/rutile_TiO2_perturbed_{i:03d}.cif"

    # Write the structure to a CIF file
    CifWriter(structure).write_file(filename, mode='wt')
    print(f"Wrote {filename}")

print("All structures generated.")
