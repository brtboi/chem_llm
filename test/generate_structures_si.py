import os
import math
import numpy as np

from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter

# SETTINGS
N_STRUCTURES = 50

BOHR_TO_ANG = 0.52917721092

os.makedirs("structures", exist_ok=True)

# DIAMOND CUBIC SILICON PARAMETERS
# Lattice constant in Angstroms
LATTICE_CONSTANT = 5.431

# Fractional coordinates of 8 atoms in diamond cubic structure
SI_FRACTIONAL_COORDS = [
    [0.0, 0.0, 0.0],
    [0.25, 0.25, 0.25],
    [0.5, 0.5, 0.0],
    [0.75, 0.75, 0.25],
    [0.5, 0.0, 0.5],
    [0.75, 0.25, 0.75],
    [0.0, 0.5, 0.5],
    [0.25, 0.75, 0.75]
]

# Species list (all Si atoms)
ATOMS = ["Si"] * 8

# Random displacement parameters (in Angstroms)
DISPLACEMENT_SIGMA = 0.05  # Gaussian sigma for atomic displacements

# Random strain range
STRAIN_EPS_MIN = -0.01
STRAIN_EPS_MAX = 0.01

# Build a single fixed diamond cubic Si structure
def build_structure():
    # Define lattice vectors based on cubic lattice with given constant
    basis_vectors = LATTICE_CONSTANT * np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # Create structure with fractional coordinates
    lattice = Lattice(basis_vectors)
    structure = Structure(
        lattice=lattice,
        species=ATOMS,
        coords=SI_FRACTIONAL_COORDS,
        coords_are_cartesian=False
    )

    return structure

# Apply random displacements and small strain
def randomize_structure(s):
    s = s.copy()

    # Apply Gaussian displacements to each atom
    for i in range(len(s)):
        dr = np.random.normal(scale=DISPLACEMENT_SIGMA, size=3)
        s.translate_sites(i, dr, frac_coords=False)

    # Apply small random strain
    eps = np.random.uniform(STRAIN_EPS_MIN, STRAIN_EPS_MAX, size=3)
    strain = np.diag(1 + eps)
    new_matrix = strain @ s.lattice.matrix

    # Recreate structure with new lattice
    structure = Structure(
        lattice=Lattice(new_matrix),
        species=s.species,
        coords=s.cart_coords,
        coords_are_cartesian=True
    )

    return structure

# Generate dataset
all_scale_pars = []
for n in range(N_STRUCTURES):
    # Use fixed par = 0 since no interpolation
    par = 0.0
    all_scale_pars.append(par)

    # Build base structure
    s = build_structure()

    # Apply randomization
    s = randomize_structure(s)

    # Save CIF file
    filename = f"structures/structure_{n:03d}.cif"
    CifWriter(s).write_file(filename, mode='wt')
    print(f"Wrote {filename}")

# Save scale_pars.dat
par_file = open("structures/scale_pars.dat", "w")
for n, par in enumerate(all_scale_pars):
    par_file.write(f"{n:03d} {par:.4f}\n")
par_file.write("\n")
par_file.close()
