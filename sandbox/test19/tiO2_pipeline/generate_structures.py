import os
import numpy as np

from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter

# SETTINGS
N_PERTURBED = 50  # number of perturbed structures to generate (51 total including base)

# Room-temperature thermal displacement parameters (in Angstrom)
# Based on Debye-Waller factors for rutile TiO2 at 300 K
# Ti: ~0.05 Angstrom, O: ~0.10 Angstrom (O is lighter, larger thermal motion)
THERMAL_DISPLACEMENT = {
    "Ti": 0.05,
    "O": 0.10
}

# Small random strain to simulate thermal expansion/contraction
MAX_STRAIN = 0.005  # 0.5% maximum strain

# Use script-relative paths so the script works regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STRUCTURES_DIR = os.path.join(SCRIPT_DIR, "structures")

os.makedirs(STRUCTURES_DIR, exist_ok=True)


def load_base_structure():
    """Load the base rutile TiO2 structure from CIF file."""
    base_cif = os.path.join(STRUCTURES_DIR, "rutile_TiO2_base.cif")
    structure = Structure.from_file(base_cif)
    return structure


def randomize_structure(s, seed=None):
    """
    Apply chemically reasonable random displacements to simulate
    room-temperature thermal vibrations.
    
    Each atom is displaced by a Gaussian random vector with standard
    deviation equal to the thermal displacement parameter for that species.
    A small random strain is also applied to simulate thermal expansion.
    """
    if seed is not None:
        np.random.seed(seed)
    
    s = s.copy()
    
    # Apply Gaussian displacements to each atom
    for i, site in enumerate(s):
        sigma = THERMAL_DISPLACEMENT[site.specie.symbol]
        dr = np.random.normal(scale=sigma, size=3)
        s.translate_sites(i, dr, frac_coords=False)
    
    # Apply small random strain (diagonal, to preserve orthogonality)
    eps = np.random.uniform(-MAX_STRAIN, MAX_STRAIN, size=3)
    strain = np.diag(1 + eps)
    new_matrix = strain @ s.lattice.matrix
    
    # Create a new Lattice from the strained matrix
    new_lattice = Lattice(new_matrix)
    
    s = Structure(
        lattice=new_lattice,
        species=s.species,
        coords=s.cart_coords,
        coords_are_cartesian=True
    )
    
    return s


# GENERATE DATASET
base_structure = load_base_structure()

# Write the base structure as structure_000.cif
base_filename = os.path.join(STRUCTURES_DIR, "structure_000.cif")
CifWriter(base_structure).write_file(base_filename, mode='wt')
print("Wrote", base_filename)

# Generate perturbed structures
for n in range(1, N_PERTURBED + 1):
    s = randomize_structure(base_structure, seed=n)
    filename = os.path.join(STRUCTURES_DIR, f"structure_{n:03d}.cif")
    CifWriter(s).write_file(filename, mode='wt')
    print("Wrote", filename)

print(f"\nGenerated {N_PERTURBED + 1} total structures (1 base + {N_PERTURBED} perturbed)")
