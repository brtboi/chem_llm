import os
import numpy as np

from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

# SETTINGS
N_STRUCTURES = 51  # 1 pristine + 50 perturbed

# Room-temperature displacement amplitudes (sigma, in Angstrom).
# These are small compared to Ti-O bond lengths (~1.95 A) and represent
# typical thermal vibration amplitudes at 300 K. Lighter O atoms vibrate
# more than heavier Ti atoms, so sigma_O > sigma_Ti.
DISPLACEMENT_SIGMA = {
    "Ti": 0.01,
    "O": 0.02,
}

os.makedirs("structures", exist_ok=True)

# Load the base rutile TiO2 structure (mp-2657, P4_2/mnm, 2 Ti + 4 O)
base_structure = Structure.from_file("structures/rutile_TiO2_base.cif")

print(f"Base structure: {base_structure.composition}, {len(base_structure)} atoms")
print(f"Lattice: a={base_structure.lattice.a:.4f}, b={base_structure.lattice.b:.4f}, c={base_structure.lattice.c:.4f}")


def randomize_structure(s, seed=None):
    """Apply random Gaussian displacements to each atom.
    
    The displacement amplitude is element-specific, reflecting the fact
    that lighter atoms have larger thermal vibration amplitudes at a given
    temperature (mean-square displacement ~ 1/sqrt(mass)).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    s = s.copy()
    
    for i, site in enumerate(s):
        sigma = DISPLACEMENT_SIGMA[site.specie.symbol]
        dr = rng.normal(scale=sigma, size=3)
        s.translate_sites(i, dr, frac_coords=False)
    
    return s


# Generate 51 structures: structure_000 = pristine, structure_001-050 = perturbed
for n in range(N_STRUCTURES):
    if n == 0:
        s = base_structure.copy()
    else:
        s = randomize_structure(base_structure, seed=n)
    
    filename = f"structures/structure_{n:03d}.cif"
    CifWriter(s).write_file(filename, mode='wt')
    print(f"Wrote {filename}")

print(f"\nDone. Generated {N_STRUCTURES} structures.")
