import os
import numpy as np

from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter

# SETTINGS
N_STRUCTURES = 51  # 1 pristine + 50 perturbed

# Room-temperature thermal displacement amplitudes (standard deviation, in Angstrom).
# O is lighter than Ti, so it has larger thermal displacement amplitude.
# These values are consistent with typical Debye-Waller factors for rutile TiO2 at 300 K.
DISPLACEMENT_MAP = {
    "Ti": 0.015,
    "O":  0.025,
}

# Small random strain to mimic thermal expansion fluctuations
STRAIN_RANGE = 0.005

# Use paths relative to this script's directory so it works regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_CIF = os.path.join(SCRIPT_DIR, "structures", "rutile_TiO2_base.cif")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "structures")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the pristine rutile TiO2 structure from the Materials Project CIF
base_structure = Structure.from_file(BASE_CIF)
print(f"Base structure: {base_structure.composition}, {len(base_structure)} atoms")
print(f"Lattice: a={base_structure.lattice.a:.4f}, b={base_structure.lattice.b:.4f}, c={base_structure.lattice.c:.4f}")


def randomize_structure(s, seed=None):
    """Apply chemically reasonable random displacements to each atom.
    
    Each atom is displaced by a Gaussian random vector with element-specific
    standard deviation (sigma). This mimics thermal vibrations at room temperature.
    A small random diagonal strain is also applied to mimic thermal expansion.
    """
    if seed is not None:
        np.random.seed(seed)
    
    s = s.copy()
    
    # Random atomic displacements (in Cartesian coordinates, Angstrom)
    for i, site in enumerate(s):
        sigma = DISPLACEMENT_MAP[site.specie.symbol]
        dr = np.random.normal(scale=sigma, size=3)
        s.translate_sites(i, dr, frac_coords=False)
    
    # Small random diagonal strain (thermal expansion fluctuation)
    eps = np.random.uniform(-STRAIN_RANGE, STRAIN_RANGE, size=3)
    strain = np.diag(1.0 + eps)
    new_matrix = strain @ s.lattice.matrix
    
    # Create a new Lattice object from the strained matrix
    s = Structure(
        lattice=Lattice(new_matrix),
        species=s.species,
        coords=s.cart_coords,
        coords_are_cartesian=True
    )
    
    return s


# Generate the dataset
for n in range(N_STRUCTURES):
    if n == 0:
        # Structure 000 is the pristine (unperturbed) rutile TiO2
        s = base_structure.copy()
    else:
        # Structures 001-050 are perturbed with random displacements
        s = randomize_structure(base_structure, seed=n)
    
    filename = os.path.join(OUTPUT_DIR, f"structure_{n:03d}.cif")
    CifWriter(s).write_file(filename, mode='wt')
    print(f"Wrote {filename}")

print(f"\nGenerated {N_STRUCTURES} structures total.")
