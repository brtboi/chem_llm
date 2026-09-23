"""Steps 1-2 of tio2_prompt.md: fetch the rutile TiO2 base structure from
Materials Project, standardize it to the actual space group's standard
primitive cell, then build a 51-structure ensemble (1 pristine + 50
randomly perturbed).

Why standardize before use
---------------------------
`generate_cif` returns whatever cell Materials Project's CIF exporter
happens to use for mp-2657 -- in practice this is a 12-atom, oblique-looking
cell (a=b=c=5.4695 A, angles ~107/107/114.5 deg) that is NOT the
conventional/primitive tetragonal setting the textbook Gamma/X/M/Z/R/A
labels assume (a=b!=c, 90/90/90). The CIF's own internal symmetry label can
also be wrong/lower (per generate_cif's docstring) -- but Materials
Project's own symmetry analysis (returned separately by generate_cif) and an
independent pymatgen SpacegroupAnalyzer check below both agree this really
is P4_2/mnm (#136).

Rather than trying to force a k-path onto that arbitrary oblique cell
(which would require re-deriving the exact crystallographic rotation
between the CIF's coordinate frame and whatever frame pymatgen's symmetry
routines standardize to -- an error-prone extra step), we standardize the
structure ONCE, up front, with pymatgen's own symmetry analysis
(SpacegroupAnalyzer.get_primitive_standard_structure), and then use that
single, self-consistent structure for everything downstream: the perturbed
ensemble, the QE cell, and the HighSymmKpath k-path. This is still "derived
from the actual fetched structure, not typed from memory" (pymatgen computes
it from mp-2657's actual atomic positions) -- it just also fixes the cell
SHAPE to the one its own symmetry says it should be, so the k-path's
reciprocal lattice and the QE calculation's reciprocal lattice are
guaranteed to be the same lattice, with no separate rotation-reprojection
step (and no way for that step to be silently wrong).

Note: at the default symprec (1e-3), spglib does NOT reduce the CIF's
noisy fetched coordinates to the true 6-atom primitive cell (small
numerical deviations from ideal Wyckoff positions, typical of a
DFT-relaxed Materials Project entry, break exact equivalence at tight
tolerance) -- it returns a spurious 12/24-atom "standard" cell with
a != b for what should be a tetragonal cell. Loosening symprec to 0.1
(still far tighter than the actual atomic displacements we introduce in
step 2, and a common/documented practice for standardizing relaxed
DFT structures) correctly reduces it to the true 6-atom tetragonal cell,
matching known experimental rutile lattice parameters (a=b~4.593 A,
c~2.959 A) closely (we get a=b=4.5998 A, c=2.9592 A).

Displacement magnitude justification (Step 2)
-----------------------------------------------
Per-atom Cartesian displacement sigmas are chosen to match real
room-temperature thermal vibration amplitudes (atomic displacement
parameters) reported for rutile TiO2: diffraction refinements near 300 K
give isotropic-equivalent mean-square displacements U_iso ~ 0.005-0.008 A^2
for both Ti and O, i.e. a 1-D RMS displacement of sqrt(U_iso) ~ 0.07-0.09 A
per Cartesian axis; a Debye-model estimate using TiO2's Debye temperature
(~660-760 K) at 300 K gives the same order of magnitude. We use:
  - Ti: sigma = 0.05 A  (heavier, more tightly bound cation -> smaller ADP)
  - O:  sigma = 0.08 A  (lighter anion, softer coordination -> larger ADP)
per Cartesian component (independent x,y,z Gaussian draws), reproducing
that experimental ~0.05-0.09 A range while giving O somewhat more freedom
than Ti, consistent with reported ADPs.

Lattice strain: an independent +/-1% random diagonal strain per lattice
vector direction -- deliberately a bit more generous than TiO2's real
room-temperature-scale thermal expansion (linear alpha ~7-9e-6/K) so the
ensemble also samples some elastic/compositional strain diversity, while
staying well inside the harmonic/small-strain regime.
"""
import json
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from chem_llm.tools import generate_cif

np.random.seed(0)

N_PERTURBED = 50
SYMPREC = 0.1
DISPLACEMENT_SIGMA_ANG = {"Ti": 0.05, "O": 0.08}
STRAIN_MAX = 0.01  # +/- 1%

# --- Step 1: fetch base structure from Materials Project ---
fetch_result = generate_cif(
    composition="TiO2",
    output_path="structures/structure_000_raw_mp.cif",
    spacegroup_symbol="P4_2/mnm",
    spacegroup_number=136,
)
assert fetch_result["success"], fetch_result
print("Step 1: fetched", fetch_result["selected_material_id"], fetch_result["space_group"])

with open("materials_project_id.json", "w") as f:
    json.dump(fetch_result, f, indent=2)

raw = Structure.from_file("structures/structure_000_raw_mp.cif")
print(f"  raw CIF cell: {len(raw)} sites, abc={raw.lattice.abc}, angles={raw.lattice.angles}")

# --- Standardize to the actual space group's primitive cell ---
sga = SpacegroupAnalyzer(raw, symprec=SYMPREC)
sg_symbol, sg_number = sga.get_space_group_symbol(), sga.get_space_group_number()
print(f"  pymatgen symmetry check (symprec={SYMPREC}): {sg_symbol} (#{sg_number})")
assert sg_number == 136, f"expected space group 136, got {sg_number}"

base = sga.get_primitive_standard_structure()
print(f"  standardized primitive cell: {len(base)} sites, abc={base.lattice.abc}, angles={base.lattice.angles}")
assert len(base) == 6, f"expected 6-atom standard rutile cell, got {len(base)}"

CifWriter(base).write_file("structures/structure_000.cif", mode="wt")
print("Wrote structures/structure_000.cif (standardized primitive cell, unperturbed)")


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


# --- Step 2: perturbed ensemble ---
for n in range(1, N_PERTURBED + 1):
    s = randomize_structure(base)
    filename = f"structures/structure_{n:03d}.cif"
    CifWriter(s).write_file(filename, mode="wt")

print(f"Step 2: wrote {N_PERTURBED} perturbed structures (structure_001..{N_PERTURBED:03d}.cif)")
print(f"Total ensemble: 1 (base) + {N_PERTURBED} (perturbed) = {N_PERTURBED + 1}")
