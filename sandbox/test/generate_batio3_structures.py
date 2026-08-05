import os
import math
import numpy as np

from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter

# SETTINGS
N_STRUCTURES = 50

BOHR_TO_ANG = 0.52917721092

os.makedirs("structures", exist_ok=True)

# REFERENCE INTERPOLATION: BaTiO3
# Cubic phase (Pm-3m): ideal perovskite, a = 4.00 Å
# Tetragonal phase (I4/mcm): a ≈ b ≈ 3.99 Å, c ≈ 4.04 Å (experimental, ~100 K)
# Ti off-centering along c-axis, O atoms displaced in-plane

# Lattice scale factors (in Bohr)
cubicScale = 4.00 / BOHR_TO_ANG  # ~7.548 Bohr
orthoScale = 4.04 / BOHR_TO_ANG  # ~7.624 Bohr

# Lattice parameters (dimensionless, scaled by cubicScale)
cubicA = 1.0
orthogonalA = 3.99 / 4.00  # a ≈ 3.99 Å
orthogonalC = 4.04 / 4.00  # c ≈ 4.04 Å

# Interpolation: cubic (par=0) → tetragonal (par=1)
def build_structure(par):
    # Scale lattice
    parScale = cubicScale * (1 - par) + orthoScale * par

    # Lattice vectors: cubic (a, a, a) → tetragonal (a, a, c)
    # Use cubic basis vectors (simple cubic) and scale accordingly
    basis_vectors = parScale * np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # Apply tetragonal distortion: scale c-axis
    basis_vectors[2] *= orthogonalC

    # Convert to Angstroms
    basis_vectors *= BOHR_TO_ANG

    # Species: 1 Ba, 1 Ti, 3 O
    atoms = ["Ba", "Ti", "O", "O", "O"]

    # Fractional coordinates
    # Cubic: Ba at (0.5, 0.5, 0.5), Ti at (0.0, 0.0, 0.0), O at (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)
    # Tetragonal: Ti off-center along c-axis, O atoms displaced in-plane
    # Distortion parameters: d1 (Ti z-displacement), d2 (O x/y displacement)
    d1 = 0.02 * par  # Ti off-centering along c-axis (0.02 Å max)
    d2 = 0.01 * par  # O displacement in-plane (0.01 Å max)

    coords = np.array([
        [0.5, 0.5, 0.5],  # Ba
        [0.0, 0.0, 0.0 + d1],  # Ti
        [0.5, 0.5, 0.0],  # O1
        [0.5, 0.0, 0.5],  # O2
        [0.0, 0.5, 0.5]   # O3
    ])

    # Apply in-plane O displacements
    coords[2, 0] += d2  # O1 x
    coords[2, 1] += d2  # O1 y
    coords[3, 0] += d2  # O2 x
    coords[3, 1] -= d2  # O2 y
    coords[4, 0] -= d2  # O3 x
    coords[4, 1] += d2  # O3 y

    coords %= 1.0

    lattice = Lattice(basis_vectors)

    structure = Structure(
        lattice=lattice,
        species=atoms,
        coords=coords,
        coords_are_cartesian=False
    )

    return structure

# RANDOM DISTORTIONS
def randomize_structure(s):
    s = s.copy()

    # Displacement map: smaller sigma due to stronger ionic bonds in BaTiO3
    displacement_map = {
        "Ba": 0.10,  # Larger ion, less mobile
        "Ti": 0.15,  # Smaller, more mobile
        "O": 0.12   # Intermediate
    }

    for i, site in enumerate(s):
        sigma = displacement_map[site.specie.symbol]
        dr = np.random.normal(scale=sigma, size=3)
        s.translate_sites(i, dr, frac_coords=False)

    # Small random strain (uniform -0.01 to +0.01)
    eps = np.random.uniform(-0.01, 0.01, size=3)
    strain = np.diag(1 + eps)
    new_matrix = strain @ s.lattice.matrix

    s = Structure(
        lattice=Lattice(new_matrix),
        species=s.species,
        coords=s.cart_coords,
        coords_are_cartesian=True
    )

    return s

# GENERATE DATASET
all_scale_pars = []
for n in range(N_STRUCTURES):
    par = np.random.uniform(0.0, 1.0)
    all_scale_pars.append(par)

    s = build_structure(par)
    s = randomize_structure(s)

    filename = f"structures/structure_{n:03d}.cif"
    CifWriter(s).write_file(filename, mode='wt')
    print(f"Wrote {filename}")

# Save scale parameters
par_file = open("structures/scale_pars.dat", "w")
for n, par in enumerate(all_scale_pars):
    par_file.write(f"{n:03d} {par:.4f}\n")
par_file.write("\n")
par_file.close()
