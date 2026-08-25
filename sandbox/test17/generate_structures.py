import os
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter

# SETTINGS
N_STRUCTURES = 50

# Ensure output directory exists
os.makedirs("structures", exist_ok=True)

# Read the base cubic CsPbBr3 structure from Materials Project CIF
base_structure = Structure.from_file("structures/cubic_CsPbBr3.cif")

# Displacement map for chemically reasonable random displacements at room temperature
# Values in Angstroms
displacement_map = {
    "Cs": 0.25,
    "Pb": 0.14,
    "Br": 0.18
}

# Generate 50 perturbed structures (total 51 including base)
for n in range(N_STRUCTURES):
    # Start with the base structure
    s = base_structure.copy()

    # Apply random displacements to each atom
    for i, site in enumerate(s):
        element = site.specie.symbol
        sigma = displacement_map[element]
        # Random displacement in Cartesian coordinates (in Angstroms)
        dr = np.random.normal(scale=sigma, size=3)
        s.translate_sites(i, dr, frac_coords=False)

    # Apply small random strain to mimic thermal fluctuations
    eps = np.random.uniform(-0.015, 0.015, size=3)
    strain = np.diag(1 + eps)
    new_matrix = strain @ s.lattice.matrix
    s = Structure(
        lattice=Lattice(new_matrix),
        species=s.species,
        coords=s.cart_coords,
        coords_are_cartesian=True
    )

    # Save the perturbed structure as a CIF file
    filename = f"structures/structure_{n:03d}.cif"
    CifWriter(s).write_file(filename, mode='wt')
    print(f"Wrote {filename}")

# Save the base structure as the first file
base_filename = "structures/structure_000.cif"
CifWriter(base_structure).write_file(base_filename, mode='wt')
print(f"Wrote {base_filename}")

print("All structures generated.")
