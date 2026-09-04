"""Build a single, unperturbed cubic CsPbBr3 structure.

Adapted from example_quantum_espresso_workflow/generate_structures.py's
build_structure(par) with par=0.0 (pure cubic endpoint of the
cubic<->orthorhombic interpolation) and WITHOUT the randomize_structure()
perturbation step -- this is the "bypass generate_structures perturbation
entirely" single base structure requested for the DeePseudopot tool-use demo.
"""
import math
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter

BOHR_TO_ANG = 0.52917721092


def build_structure(par):
    cubicScale = 11.09269 * math.sqrt(2)
    cubicA = 1.0
    cubicB = 1.0
    cubicC = math.sqrt(2)

    orthoScale = 15.594
    orthoA = 1.0
    orthoB = 0.9940620455647114
    orthoC = 1.4220794958797864

    parScale = cubicScale * (1 - par) + orthoScale * par
    parA = cubicA * (1 - par) + orthoA * par
    parB = cubicB * (1 - par) + orthoB * par
    parC = cubicC * (1 - par) + orthoC * par

    d1 = (0.25 - 0.19462) * par
    d2 = (0.25 - 0.19731) * par
    d3 = 0.03577 * par
    d4 = 0.00113 * par
    d5 = 0.06202 * par
    d6 = 0.04005 * par
    d7 = 0.00509 * par

    basis_vectors = parScale * np.array([
        [parA / math.sqrt(2), -parA / math.sqrt(2), 0],
        [parB / math.sqrt(2), parB / math.sqrt(2), 0],
        [0, 0, parC],
    ])
    basis_vectors *= BOHR_TO_ANG

    atoms = (
        ["Cs"] * 4 + ["Pb"] * 4 + ["Br"] * 12
    )

    coords = np.array([
        [0.50 - d6, 0.5 + d7, 0.25000],
        [0.50 + d6, 0.5 - d7, 0.75000],
        [1.00 - d6, 0.0 + d7, 0.25000],
        [0.00 + d6, 1.0 - d7, 0.75000],

        [0.50000, 0.00000, 0.0000],
        [0.00000, 0.50000, 0.0000],
        [0.50000, 0.00000, 0.5000],
        [0.00000, 0.50000, 0.5000],

        [0.25 - d1, 0.25 - d2, 0.0 + d3],
        [0.75 - d1, 0.25 + d2, 0.0 + d3],
        [0.75 + d1, 0.75 + d2, 1.0 - d3],
        [0.25 + d1, 0.75 - d2, 1.0 - d3],

        [0.25 - d1, 0.25 - d2, 0.5 - d3],
        [0.75 - d1, 0.25 + d2, 0.5 - d3],
        [0.75 + d1, 0.75 + d2, 0.5 + d3],
        [0.25 + d1, 0.75 - d2, 0.5 + d3],

        [0.00 + d4, 0.50 + d5, 0.2500],
        [0.50 + d4, 1.00 - d5, 0.2500],
        [1.00 - d4, 0.50 - d5, 0.7500],
        [0.50 - d4, 0.00 + d5, 0.7500],
    ])
    coords %= 1.0

    lattice = Lattice(basis_vectors)
    return Structure(lattice=lattice, species=atoms, coords=coords, coords_are_cartesian=False)


if __name__ == "__main__":
    # par=0.0 -> pure cubic endpoint, no random thermal displacement applied
    # (randomize_structure() from generate_structures.py is intentionally
    # skipped: this is the single, clean base structure for the pipeline).
    s = build_structure(0.0)
    print(s)
    print("lattice abc:", s.lattice.abc)
    print("lattice angles:", s.lattice.angles)
    CifWriter(s).write_file("/pscratch/sd/b/brenthu/chem_llm/deepseudopot_example/structure_cubic_CsPbBr3.cif", mode="wt")
    print("wrote structure_cubic_CsPbBr3.cif")
