import os
import math
import numpy as np

from ase import Atoms
from ase.io import write

# SETTINGS
N_STRUCTURES = 50

BOHR_TO_ANG = 0.52917721092

os.makedirs("structures", exist_ok=True)

# REFERENCE INTERPOLATION
def build_structure(par):

    # lattice interpolation (CsPbBr3, from bromideMixBuild.py)
    cubicScale = 11.09269 * math.sqrt(2)

    cubicA = 1.0
    cubicB = 1.0
    cubicC = math.sqrt(2)

    orthoScale = 15.594

    orthoA = 1.0
    orthoB = 0.9940620455647114
    orthoC = 1.4220794958797864

    parScale = cubicScale*(1-par) + orthoScale*par

    parA = cubicA*(1-par) + orthoA*par
    parB = cubicB*(1-par) + orthoB*par
    parC = cubicC*(1-par) + orthoC*par

    # internal distortion interpolation
    d1 = (0.25-0.19462)*par
    d2 = (0.25-0.19731)*par
    d3 = 0.03577*par
    d4 = 0.00113*par
    d5 = 0.06202*par
    d6 = 0.04005*par
    d7 = 0.00509*par

    # lattice vectors
    basis_vectors = parScale * np.array([
        [ parA/math.sqrt(2), -parA/math.sqrt(2), 0 ],
        [ parB/math.sqrt(2),  parB/math.sqrt(2), 0 ],
        [ 0, 0, parC ]
    ])

    basis_vectors *= BOHR_TO_ANG

    # species
    atoms = [
        "Cs","Cs","Cs","Cs",
        "Pb","Pb","Pb","Pb",
        "Br","Br","Br","Br",
        "Br","Br","Br","Br",
        "Br","Br","Br","Br"
    ]

    # fractional coordinates
    coords = np.array([

        [0.50-d6, 0.5+d7, 0.25000],
        [0.50+d6, 0.5-d7, 0.75000],
        [1.00-d6, 0.0+d7, 0.25000],
        [0.00+d6, 1.0-d7, 0.75000],

        [0.50000, 0.00000, 0.0000],
        [0.00000, 0.50000, 0.0000],
        [0.50000, 0.00000, 0.5000],
        [0.00000, 0.50000, 0.5000],

        [0.25-d1, 0.25-d2, 0.0+d3],
        [0.75-d1, 0.25+d2, 0.0+d3],
        [0.75+d1, 0.75+d2, 1.0-d3],
        [0.25+d1, 0.75-d2, 1.0-d3],

        [0.25-d1, 0.25-d2, 0.5-d3],
        [0.75-d1, 0.25+d2, 0.5-d3],
        [0.75+d1, 0.75+d2, 0.5+d3],
        [0.25+d1, 0.75-d2, 0.5+d3],

        [0.00+d4, 0.50+d5, 0.2500],
        [0.50+d4, 1.00-d5, 0.2500],
        [1.00-d4, 0.50-d5, 0.7500],
        [0.50-d4, 0.00+d5, 0.7500]

    ])

    coords %= 1.0

    structure = Atoms(
        symbols=atoms,
        cell=basis_vectors,
        scaled_positions=coords,
        pbc=True,
    )

    return structure

# RANDOM DISTORTIONS
def randomize_structure(s):

    s = s.copy()

    displacement_map = {
        "Cs": 0.25,
        "Pb": 0.14,
        "Br": 0.18
    }

    # Cartesian (Angstrom) per-atom displacement, drawn per-species -- ASE
    # positions are Cartesian by default, so this translates each atom
    # directly (equivalent to the old translate_sites(..., frac_coords=False)).
    for i, symbol in enumerate(s.get_chemical_symbols()):

        sigma = displacement_map[symbol]

        dr = np.random.normal(scale=sigma, size=3)

        s.positions[i] += dr

    # small random strain

    eps = np.random.uniform(-0.015, 0.015, size=3)

    strain = np.diag(1 + eps)

    new_cell = strain @ s.cell[:]

    # scale_atoms=False keeps the just-applied Cartesian positions fixed and
    # only changes the cell -- matches rebuilding the old Structure with the
    # same cart_coords under a strained lattice.
    s.set_cell(new_cell, scale_atoms=False)

    return s

ANG_TO_BOHR = 1.889726125

# GENERATE DATASET
all_scale_pars = []
for n in range(N_STRUCTURES):

    # interpolate between cubic and ortho

    par = np.random.uniform(0.0, 1.0)
    all_scale_pars.append(par)

    s = build_structure(par)

    s = randomize_structure(s)

    filename = f"structures/structure_{n:03d}.cif"

    # Geometry
    write(filename, s, format="cif")

    print("Wrote", filename)

par_file = open("structures/scale_pars.dat", "w")
for n, par in enumerate(all_scale_pars):
    par_file.write(f"{n:03d} {par:.4f}\n")

par_file.write("\n")
par_file.close()
