"""Step 2: build a 51-structure ensemble for rutile TiO2 (mp-2657).

structure_000.cif is the unperturbed Materials Project structure (fetched by
step1_fetch_structure.py). structure_001..050.cif are copies of it with
chemically-reasonable random atomic displacements (room-temperature thermal
vibration amplitudes) plus a small random lattice strain.

Displacement magnitude justification
-------------------------------------
We want per-atom Cartesian displacement sigmas comparable to real
room-temperature thermal vibration amplitudes (atomic displacement
parameters, ADPs) in rutile TiO2, not an arbitrary number. Neutron/X-ray
diffraction refinements of rutile TiO2 near 300 K report isotropic-equivalent
mean-square displacements U_iso ~ 0.005-0.008 A^2 for both Ti and O
(e.g. Burdett et al., Restori & Schwarzenbach-type rutile refinements), i.e.
a 1-D RMS displacement of sqrt(U_iso) ~ 0.07-0.09 A per Cartesian axis. A
simple Debye-model estimate using TiO2's Debye temperature (~660-760 K) at
300 K gives the same order of magnitude (~0.05-0.08 A). We use:
  - Ti: sigma = 0.05 A  (heavier, more tightly bound cation -> smaller ADP)
  - O:  sigma = 0.08 A  (lighter anion, softer octahedral coordination
        environment -> larger ADP)
per Cartesian component (independent x,y,z Gaussian draws), which reproduces
that experimental 0.05-0.09 A range while giving O somewhat more freedom
than Ti, consistent with reported ADPs.

Lattice strain: an independent +/-1% random diagonal strain per lattice
vector direction, representing typical thermal-expansion-scale lattice
fluctuations (linear thermal expansion coefficient of rutile TiO2 is
~7-9e-6/K; +/-1% is deliberately generous compared to a real T-linked value,
to also sample some elastic/compositional strain diversity for the training
ensemble, without leaving the harmonic/small-strain regime).
"""
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter

np.random.seed(0)

N_PERTURBED = 50

DISPLACEMENT_SIGMA_ANG = {
    "Ti": 0.05,
    "O": 0.08,
}
STRAIN_MAX = 0.01  # +/- 1%

base = Structure.from_file("structures/structure_000_raw.cif")
CifWriter(base).write_file("structures/structure_000.cif", mode="wt")
print("Wrote structures/structure_000.cif (unperturbed, mp-2657)")


def randomize_structure(s: Structure) -> Structure:
    s = s.copy()

    for i, site in enumerate(s):
        sigma = DISPLACEMENT_SIGMA_ANG[site.specie.symbol]
        dr = np.random.normal(scale=sigma, size=3)
        s.translate_sites(i, dr, frac_coords=False)

    eps = np.random.uniform(-STRAIN_MAX, STRAIN_MAX, size=3)
    strain = np.diag(1 + eps)
    new_matrix = strain @ s.lattice.matrix

    return Structure(
        lattice=Lattice(new_matrix),
        species=s.species,
        coords=s.cart_coords,
        coords_are_cartesian=True,
    )


for n in range(1, N_PERTURBED + 1):
    s = randomize_structure(base)
    filename = f"structures/structure_{n:03d}.cif"
    CifWriter(s).write_file(filename, mode="wt")
    print("Wrote", filename)

print(f"\nTotal structures: 1 (base) + {N_PERTURBED} (perturbed) = {N_PERTURBED + 1}")
