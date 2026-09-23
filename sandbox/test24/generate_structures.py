import os
import numpy as np

from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter

# SETTINGS
N_STRUCTURES = 50  # number of perturbed structures (51 total including base)

# Displacement sigmas (in Angstrom) for room-temperature thermal vibrations.
# O is lighter (16 amu) than Ti (48 amu), so it has larger thermal displacement.
# Typical room-temperature Debye-Waller displacements for rutile TiO2:
#   Ti: ~0.08-0.12 Angstrom, O: ~0.12-0.18 Angstrom
DISPLACEMENT_MAP = {
    "Ti": 0.10,
    "O": 0.15,
}

# Small random strain to simulate slight lattice distortion
STRAIN_RANGE = 0.015  # +/- 1.5%

os.makedirs("structures", exist_ok=True)

# Load the base rutile TiO2 structure (mp-2657, space group P4_2/mnm No. 136)
# The CIF file reports P1 due to generate_cif's known behavior, but the
# structure itself is the primitive cell of rutile TiO2.
base_structure = Structure.from_file("structures/base_TiO2.cif")

# Verify the base structure has the expected composition and atom count
assert len(base_structure) == 12, f"Expected 12 atoms, got {len(base_structure)}"
assert base_structure.composition.reduced_formula == "TiO2", \
    f"Expected TiO2, got {base_structure.composition.reduced_formula}"

# Write the base structure as structure_000.cif
CifWriter(base_structure).write_file("structures/structure_000.cif", mode='wt')
print("Wrote structures/structure_000.cif (base rutile TiO2)")


def randomize_structure(s, seed=None):
    """Apply random displacements and small random strain to a structure.
    
    The displacements are Gaussian with element-specific sigma values that
    approximate room-temperature thermal vibrations in rutile TiO2.
    A small diagonal strain (up to +/- STRAIN_RANGE) is also applied to
    simulate slight lattice distortion.
    """
    if seed is not None:
        np.random.seed(seed)
    
    s = s.copy()
    
    # Apply random displacements to each atom
    for i, site in enumerate(s):
        sigma = DISPLACEMENT_MAP[site.specie.symbol]
        dr = np.random.normal(scale=sigma, size=3)
        s.translate_sites(i, dr, frac_coords=False)
    
    # Apply small random strain (diagonal, in Cartesian coordinates)
    eps = np.random.uniform(-STRAIN_RANGE, STRAIN_RANGE, size=3)
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


# Generate perturbed structures
for n in range(1, N_STRUCTURES + 1):
    s = randomize_structure(base_structure, seed=n)
    
    # INVARIANT 1: nat/ntyp must be computed from the actual structure object
    nat = len(s)
    ntyp = len(s.symbol_set)
    assert nat == 12, f"Structure {n:03d}: expected nat=12, got {nat}"
    assert ntyp == 2, f"Structure {n:03d}: expected ntyp=2, got {ntyp}"
    
    filename = f"structures/structure_{n:03d}.cif"
    CifWriter(s).write_file(filename, mode='wt')
    print(f"Wrote {filename} (nat={nat}, ntyp={ntyp})")

print(f"\nDone. Generated {N_STRUCTURES + 1} total structures (1 base + {N_STRUCTURES} perturbed).")
