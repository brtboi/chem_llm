import os
import numpy as np

from ase.io import read, write

# SETTINGS
N_STRUCTURES = 50

# Per-species Gaussian displacement sigma (Angstrom) for room-temperature
# phonon-like random displacements.  O is lighter than Ti, so at the same
# temperature it has a larger zero-point / thermal amplitude; these values
# are chemically reasonable for a room-temperature rutile TiO2 structure.
DISPLACEMENT_SIGMA = {
    "Ti": 0.10,
    "O":  0.15,
}

# Small random diagonal strain applied to each perturbed structure.
STRAIN_RANGE = (-0.015, 0.015)

BASE_CIF = "structures/base_TiO2.cif"
OUT_DIR = "structures"

os.makedirs(OUT_DIR, exist_ok=True)


def randomize_structure(s, rng):
    """Apply per-atom Gaussian Cartesian displacements and a small random
    diagonal strain to a copy of the input structure.

    ASE positions are Cartesian (Angstrom) by default, so adding a
    per-atom 3-vector directly displaces that atom in real space.
    set_cell(scale_atoms=False) keeps the just-displaced Cartesian
    positions fixed and only changes the cell vectors, matching the
    behaviour of the reference pipeline.
    """
    s = s.copy()

    for i, symbol in enumerate(s.get_chemical_symbols()):
        sigma = DISPLACEMENT_SIGMA[symbol]
        dr = rng.normal(scale=sigma, size=3)
        s.positions[i] += dr

    eps = rng.uniform(STRAIN_RANGE[0], STRAIN_RANGE[1], size=3)
    strain = np.diag(1.0 + eps)
    new_cell = strain @ s.cell[:]
    s.set_cell(new_cell, scale_atoms=False)

    return s


def main():
    rng = np.random.default_rng(seed=42)

    # Read the base rutile TiO2 structure (mp-2657, P4_2/mnm #136).
    # The MP entry uses a 12-atom cell (4 Ti, 8 O) -- a valid supercell of
    # the 6-atom primitive rutile cell.  We use it as-is.
    base = read(BASE_CIF)

    # INVARIANT 1: nat and ntyp must be computed from the actual ASE Atoms
    # object in hand, never hardcoded.  Record them here and use them for
    # all subsequent consistency checks.
    nat_base = len(base)
    ntyp_base = len(set(base.get_chemical_symbols()))
    print(f"Base structure: {nat_base} atoms, {ntyp_base} species, "
          f"cell = {base.cell[:].tolist()}")

    # Write the unperturbed base structure as structure_000.cif
    write(os.path.join(OUT_DIR, "structure_000.cif"), base, format="cif")
    print("Wrote structures/structure_000.cif (unperturbed base)")

    for n in range(1, N_STRUCTURES + 1):
        s = randomize_structure(base, rng)

        # INVARIANT 1: nat/ntyp must match the base structure exactly --
        # perturbation only displaces atoms and strains the cell, it does
        # not add or remove atoms or change species.
        nat = len(s)
        ntyp = len(set(s.get_chemical_symbols()))
        assert nat == nat_base, (
            f"Structure {n:03d}: expected {nat_base} atoms, got {nat}"
        )
        assert ntyp == ntyp_base, (
            f"Structure {n:03d}: expected {ntyp_base} species, got {ntyp}"
        )

        filename = os.path.join(OUT_DIR, f"structure_{n:03d}.cif")
        write(filename, s, format="cif")
        print(f"Wrote {filename}")

    print(f"Done. Wrote {N_STRUCTURES + 1} structures total.")


if __name__ == "__main__":
    main()
