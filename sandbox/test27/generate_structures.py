#!/usr/bin/env python
"""
Generate 51 rutile TiO2 structures: structure_000.cif (unperturbed base)
and structure_001.cif - structure_050.cif (perturbed copies).

Perturbation scheme (chemically reasonable for room-temperature thermal motion):
  - Per-atom Gaussian displacement: sigma = 0.05 Angstrom per Cartesian component.
    Justification: the Debye-Waller / atomic displacement parameter (ADP) for
    Ti and O in rutile TiO2 at 300 K is on the order of 0.05-0.1 Angstrom
    (see e.g. neutron diffraction studies). 0.05 A is at the low end of this
    range, small enough to preserve local bonding geometry (no bond breaking,
    no unphysical overlaps) while still sampling thermally accessible
    configurations. This is a static snapshot approximation to the
    instantaneous atomic positions at 300 K.
  - Small random lattice strain: each component of the strain tensor drawn
    uniformly from [-0.005, 0.005] (i.e. up to 0.5% strain). This mimics the
    small elastic fluctuations of the unit cell at finite temperature and
    ensures the perturbed structures are not all identical in cell shape.

No pymatgen is used anywhere in this script.
"""

import numpy as np
from ase.io import read, write
from ase import Atoms

# Fixed seed for reproducibility
rng = np.random.default_rng(seed=42)

# Displacement magnitude: 0.05 Angstrom (see module docstring for justification)
DISPLACEMENT_SIGMA = 0.05  # Angstrom
# Strain magnitude: uniform in [-0.005, 0.005] (up to 0.5% per component)
STRAIN_MAX = 0.005

BASE_CIF = "structures/TiO2_base.cif"
OUT_DIR = "structures"


def perturb_structure(atoms: Atoms, rng: np.random.Generator) -> Atoms:
    """Return a perturbed copy of `atoms` with random atomic displacements
    and a small random lattice strain.

    Crystallographic operations:
      1. Apply a small random strain tensor to the cell vectors:
         new_cell = (I + strain) @ old_cell
         This is a linear transformation of the lattice vectors, preserving
         the crystallographic cell shape up to a small distortion.
      2. Displace each atom in Cartesian coordinates by a Gaussian random
         vector with standard deviation DISPLACEMENT_SIGMA per component.
         The displacements are applied in the Cartesian frame, so they are
         isotropic in real space regardless of the cell orientation.
    """
    # Work on a copy so the original is not modified
    perturbed = atoms.copy()

    # --- Lattice strain ---
    # Random strain tensor: each of the 9 components drawn uniformly
    strain = rng.uniform(-STRAIN_MAX, STRAIN_MAX, size=(3, 3))
    # Apply strain to cell vectors: new_cell = (I + strain) @ old_cell
    # (row-vector convention: cell vectors are rows of the cell matrix)
    old_cell = np.array(perturbed.cell[:])
    new_cell = (np.eye(3) + strain) @ old_cell
    perturbed.cell = new_cell

    # --- Atomic displacements ---
    # Gaussian displacement in Cartesian coordinates
    displacements = rng.normal(0.0, DISPLACEMENT_SIGMA, size=perturbed.positions.shape)
    perturbed.positions += displacements

    return perturbed


def main():
    # Read the base rutile TiO2 structure
    base = read(BASE_CIF, format="cif")
    print(f"Base structure: {len(base)} atoms, cell = {base.cell[:].tolist()}")
    print(f"Chemical symbols: {base.get_chemical_symbols()}")

    # INVARIANT 1: nat and ntyp must be computed from the actual Atoms object,
    # never hardcoded. The base structure from mp-2657 has 12 atoms
    # (4 Ti + 8 O) -- the conventional rutile unit cell with Z=4.
    base_nat = len(base)
    base_ntyp = len(set(base.get_chemical_symbols()))
    print(f"Base nat={base_nat}, ntyp={base_ntyp}")

    # Write structure_000.cif (unperturbed base)
    write(f"{OUT_DIR}/structure_000.cif", base, format="cif")
    print(f"Wrote {OUT_DIR}/structure_000.cif (unperturbed base)")

    # Generate 50 perturbed structures
    for i in range(1, 51):
        perturbed = perturb_structure(base, rng)

        # INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
        nat = len(perturbed)
        ntyp = len(set(perturbed.get_chemical_symbols()))
        assert nat == base_nat, (
            f"Structure {i:03d}: nat changed from {base_nat} to {nat}"
        )
        assert ntyp == base_ntyp, (
            f"Structure {i:03d}: ntyp changed from {base_ntyp} to {ntyp}"
        )

        out_path = f"{OUT_DIR}/structure_{i:03d}.cif"
        write(out_path, perturbed, format="cif")
        print(f"Wrote {out_path} (nat={nat}, ntyp={ntyp})")

    print("\nDone: 51 structures written to structures/")


if __name__ == "__main__":
    main()
