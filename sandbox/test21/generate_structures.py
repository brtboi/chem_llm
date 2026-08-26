import os
import numpy as np

from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter

# SETTINGS
N_STRUCTURES = 51  # 1 pristine + 50 perturbed

# Room-temperature thermal displacement amplitudes (standard deviation, in Angstrom)
# Based on Debye-Waller factors for rutile TiO2 at 300 K:
#   Ti: ~0.06 A (heavier atom, smaller displacement)
#   O:  ~0.10 A (lighter atom, larger displacement)
DISPLACEMENT_MAP = {
    "Ti": 0.06,
    "O":  0.10,
}

# Small random strain to simulate thermal expansion/contraction (±0.5%)
MAX_STRAIN = 0.005

os.makedirs("structures", exist_ok=True)

# Load the base rutile TiO2 structure from the CIF file
# (obtained from Materials Project mp-2657, space group P4_2/mnm #136)
base_structure = Structure.from_file("structures/TiO2_rutile_base.cif")

print(f"Base structure: {base_structure.composition.formula}")
print(f"  Atoms: {len(base_structure)}")
print(f"  Lattice: a={base_structure.lattice.a:.4f}, b={base_structure.lattice.b:.4f}, c={base_structure.lattice.c:.4f}")
print(f"  Space group: {base_structure.get_space_group_info()}")


def randomize_structure(s, seed=None):
    """
    Apply chemically reasonable random displacements to each atom
    at room temperature, plus a small random strain.

    The displacements are drawn from a Gaussian distribution with
    per-element standard deviations based on Debye-Waller factors.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    s = s.copy()

    # Random atomic displacements (in Cartesian coordinates, Angstrom)
    for i, site in enumerate(s):
        sigma = DISPLACEMENT_MAP[site.specie.symbol]
        dr = rng.normal(scale=sigma, size=3)
        s.translate_sites(i, dr, frac_coords=False)

    # Small random diagonal strain (±0.5%) to simulate thermal effects
    eps = rng.uniform(-MAX_STRAIN, MAX_STRAIN, size=3)
    strain = np.diag(1.0 + eps)
    new_matrix = strain @ s.lattice.matrix

    # Create a new Lattice object from the strained matrix
    s = Structure(
        lattice=Lattice(new_matrix),
        species=s.species,
        coords=s.cart_coords,
        coords_are_cartesian=True,
    )

    return s


# Generate the dataset
# Structure 000 is the pristine (unperturbed) base structure
# Structures 001-050 are randomly perturbed
for n in range(N_STRUCTURES):
    if n == 0:
        s = base_structure.copy()
    else:
        s = randomize_structure(base_structure, seed=n)

    filename = f"structures/structure_{n:03d}.cif"
    CifWriter(s).write_file(filename, mode="wt")
    print(f"Wrote {filename}")

print(f"\nDone. Generated {N_STRUCTURES} structures in 'structures/'.")
