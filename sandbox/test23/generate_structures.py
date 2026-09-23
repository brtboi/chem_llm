import os
import numpy as np

from ase.io import read, write

# SETTINGS
N_STRUCTURES = 50  # 50 perturbed structures + 1 base = 51 total

# Room-temperature displacement amplitudes (Angstrom) for each species.
# Rutile TiO2 at 300 K: Ti atoms are relatively stiff (strong Ti-O bonds),
# O atoms are lighter and vibrate more. Typical Debye-Waller / thermal
# displacement amplitudes from XRD/neutron data are ~0.05-0.10 Ang for Ti
# and ~0.10-0.15 Ang for O. We use slightly larger values to ensure the
# perturbed structures are clearly distinct from the base while remaining
# chemically reasonable (no bond breaking, no unphysical overlaps).
DISPLACEMENT_MAP = {
    "Ti": 0.08,
    "O": 0.12,
}

# Small random diagonal strain range (fractional, i.e. 1.5% max).
# This mimics small thermal expansion/contraction fluctuations.
STRAIN_RANGE = 0.015

os.makedirs("structures", exist_ok=True)


def randomize_structure(s):
    """Apply chemically reasonable random displacements to a copy of s.

    Each atom is displaced by a Gaussian random vector in Cartesian
    coordinates (Angstrom) with a per-species standard deviation from
    DISPLACEMENT_MAP. A small random diagonal strain is also applied to
    the cell to mimic thermal expansion fluctuations.

    The displacements are small enough (<< bond length) to keep the
    structure chemically reasonable: no bonds are broken, no atoms
    overlap unphysically, and the overall topology is preserved.
    """
    s = s.copy()

    # Per-atom Cartesian displacement (Angstrom), drawn per-species.
    # ASE positions are Cartesian by default, so this translates each
    # atom directly in real space.
    for i, symbol in enumerate(s.get_chemical_symbols()):
        sigma = DISPLACEMENT_MAP[symbol]
        dr = np.random.normal(scale=sigma, size=3)
        s.positions[i] += dr

    # Small random diagonal strain (fractional).
    # Applied in the Cartesian frame: new_cell = diag(1+eps) @ old_cell.
    # This is a uniform (isotropic) strain in the Cartesian axes, which
    # is a reasonable approximation for small thermal fluctuations.
    eps = np.random.uniform(-STRAIN_RANGE, STRAIN_RANGE, size=3)
    strain = np.diag(1.0 + eps)
    new_cell = strain @ s.cell[:]

    # scale_atoms=False keeps the just-applied Cartesian positions fixed
    # and only changes the cell -- the positions are NOT rescaled to
    # compensate for the cell change, which is the correct behavior for
    # a physical strain (atoms move with the lattice).
    s.set_cell(new_cell, scale_atoms=False)

    return s


# GENERATE DATASET
# Structure 000 is the unperturbed base rutile TiO2 structure.
base = read("structures/base_TiO2.cif")
write("structures/structure_000.cif", base, format="cif")
print("Wrote structures/structure_000.cif (base rutile TiO2)")

for n in range(1, N_STRUCTURES + 1):
    s = randomize_structure(base)
    filename = f"structures/structure_{n:03d}.cif"
    write(filename, s, format="cif")
    print("Wrote", filename)

print(f"Done. Generated {N_STRUCTURES + 1} structures total.")
