import os
import numpy as np

from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter

# SETTINGS
N_STRUCTURES = 51  # 1 pristine + 50 perturbed

# Room-temperature thermal displacement amplitudes (standard deviation, in Angstrom)
# Based on Debye-Waller factors for rutile TiO2 at 300 K:
#   Ti: ~0.06 A, O: ~0.10 A
DISPLACEMENT_MAP = {
    "Ti": 0.06,
    "O": 0.10,
}

# Small random strain to account for thermal expansion fluctuations
# Rutile TiO2 has a small thermal expansion coefficient; ±0.5% is reasonable
STRAIN_RANGE = 0.005

os.makedirs("structures", exist_ok=True)

# Load the base rutile TiO2 structure from the CIF file
# The CIF is in a non-standard P1 setting with 12 atoms (4 Ti, 8 O)
# This is a valid representation of the rutile structure
base_structure = Structure.from_file("structures/TiO2_rutile_base.cif")

print(f"Base structure: {len(base_structure)} atoms, {base_structure.composition}")
print(f"Lattice: a={base_structure.lattice.a:.4f}, b={base_structure.lattice.b:.4f}, c={base_structure.lattice.c:.4f}")
print(f"Angles: alpha={base_structure.lattice.alpha:.2f}, beta={base_structure.lattice.beta:.2f}, gamma={base_structure.lattice.gamma:.2f}")


def randomize_structure(s, seed=None):
    """
    Apply random room-temperature displacements to each atom.
    
    The displacements are drawn from a Gaussian distribution with standard
    deviation equal to the thermal displacement amplitude for each element.
    A small random strain is also applied to account for thermal expansion.
    """
    if seed is not None:
        np.random.seed(seed)
    
    s = s.copy()
    
    # Apply random displacements to each atom
    for i, site in enumerate(s):
        sigma = DISPLACEMENT_MAP[site.specie.symbol]
        dr = np.random.normal(scale=sigma, size=3)
        s.translate_sites(i, dr, frac_coords=False)
    
    # Apply small random strain (thermal expansion)
    eps = np.random.uniform(-STRAIN_RANGE, STRAIN_RANGE, size=3)
    strain = np.diag(1 + eps)
    new_matrix = strain @ s.lattice.matrix
    
    # Create a new Lattice from the strained matrix
    s = Structure(
        lattice=Lattice(new_matrix),
        species=s.species,
        coords=s.cart_coords,
        coords_are_cartesian=True
    )
    
    return s


# Generate the dataset
# Structure 000 is the pristine (unperturbed) base structure
# Structures 001-050 are perturbed with random displacements

# Write the pristine structure
CifWriter(base_structure).write_file("structures/structure_000.cif", mode='wt')
print("Wrote structures/structure_000.cif (pristine)")

# Write the perturbed structures
for n in range(1, N_STRUCTURES):
    s = randomize_structure(base_structure, seed=n)
    filename = f"structures/structure_{n:03d}.cif"
    CifWriter(s).write_file(filename, mode='wt')
    print(f"Wrote {filename}")

print(f"\nGenerated {N_STRUCTURES} structures total.")
