import numpy as np
from ase.io import read, write

# Read reduced primitive rutile TiO2 (6 atoms: Ti2 O4)
base = read('structures/base_primitive.cif')
nat_expected = len(base)
ntyp_expected = len(set(base.get_chemical_symbols()))

# Displacement magnitude justification:
# Room-temperature RMS thermal displacement in oxides is ~0.07-0.1 A per
# Cartesian component (typical Debye-Waller <u^2> ~ 0.005-0.01 A^2 for O/Ti).
# We use sigma = 0.08 A per Cartesian component as a chemically reasonable
# room-temperature isotropic Gaussian displacement.
DISP_SIGMA = 0.08  # angstrom, per Cartesian component
# Small random lattice strain ~1% (Gaussian, symmetric strain tensor).
STRAIN_SIGMA = 0.01

rng = np.random.default_rng(12345)

for i in range(51):
    atoms = base.copy()
    if i > 0:
        # Random atomic displacements (thermal-like Gaussian noise)
        disp = rng.normal(0.0, DISP_SIGMA, size=(len(atoms), 3))
        atoms.set_positions(atoms.get_positions() + disp)
        # Small random symmetric lattice strain applied to the cell.
        strain = rng.normal(0.0, STRAIN_SIGMA, size=(3, 3))
        strain = 0.5 * (strain + strain.T)  # symmetrize
        defmat = np.eye(3) + strain
        # Deform cell; scale_atoms keeps fractional coords so atoms move with cell.
        new_cell = atoms.cell[:] @ defmat.T
        atoms.set_cell(new_cell, scale_atoms=True)

    # INVARIANT 1: nat/ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == nat_expected, f'nat {nat} != {nat_expected}'
    assert ntyp == ntyp_expected, f'ntyp {ntyp} != {ntyp_expected}'

    write(f'structures/structure_{i:03d}.cif', atoms)

print(f'Wrote 51 structures, nat={nat_expected}, ntyp={ntyp_expected}')
