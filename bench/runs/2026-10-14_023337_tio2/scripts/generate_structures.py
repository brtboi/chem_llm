#!/usr/bin/env python3
"""
Generate 51 perturbed TiO2 (rutile) structures for DFT band-structure calculations.

Structure 000: unperturbed primitive cell (6 atoms, P4_2/mnm #136).
Structures 001-050: random atomic displacements (thermal motion) + small random lattice strain.

Temperature choice: T = 300 K
  Rutile TiO2 is the thermodynamically stable phase of TiO2 at room temperature
  and remains stable up to ~1800 C (above which it is still rutile; the phase
  transition to a higher-symmetry phase occurs only at very high T/pressure).
  300 K is therefore a physically meaningful temperature for this phase.

Displacement amplitudes (Einstein model, per species):
  The mean-square displacement U (in A^2) is related to the RMS displacement
  by RMS = sqrt(U).  We assign per-species U values representative of
  300 K thermal motion in rutile TiO2:

    O:  U_O  = 0.008 A^2  ->  RMS_O  = sqrt(0.008) = 0.0894 A
    Ti: U_Ti = 0.004 A^2  ->  RMS_Ti = sqrt(0.004) = 0.0632 A

  Justification: O is lighter (16 g/mol vs 47.87 g/mol) and in rutile the
  O atoms occupy the 4f site with a somewhat softer local potential than the
  Ti 2a site, so O has a larger mean-square displacement.  These values are
  consistent with typical experimental U values for rutile at 300 K
  (U_O ~ 0.005-0.010 A^2, U_Ti ~ 0.003-0.006 A^2).

Lattice strain: small random strain tensor with components drawn uniformly
  from [-0.005, +0.005] (i.e. up to 0.5% strain per component).

Units: all displacements are in Angstrom (A), consistent with ASE's default.
"""

import numpy as np
import ase.io
from ase import Atoms

# --- Read the primitive cell (6 atoms) as the base structure ---
base = ase.io.read('structures/prim_TiO2.cif')
print(f"Base structure: {len(base)} atoms, cell = {base.cell}")

# --- Per-species RMS displacement (A) ---
# U in A^2, RMS = sqrt(U) in A
U_O  = 0.008   # A^2  -> RMS = 0.0894 A
U_Ti = 0.004   # A^2  -> RMS = 0.0632 A
rms_O  = np.sqrt(U_O)   # 0.0894 A
rms_Ti = np.sqrt(U_Ti)  # 0.0632 A

# --- Random lattice strain: max 0.5% per component ---
max_strain = 0.005  # 0.5%

# --- Fixed seed for reproducibility ---
np.random.seed(42)

# --- Write structure 000: unperturbed ---
ase.io.write('structures/structure_000.cif', base)
print("Wrote structures/structure_000.cif (unperturbed)")

# --- Write structures 001-050: perturbed ---
for i in range(1, 51):
    atoms = base.copy()

    # 1) Random atomic displacements (Cartesian, in A)
    #    Each atom gets a Gaussian displacement with species-specific RMS.
    disp = np.zeros((len(atoms), 3))
    for idx, sym in enumerate(atoms.get_chemical_symbols()):
        if sym == 'O':
            disp[idx] = np.random.normal(0.0, rms_O, size=3)
        elif sym == 'Ti':
            disp[idx] = np.random.normal(0.0, rms_Ti, size=3)
        else:
            raise ValueError(f"Unexpected species {sym}")
    atoms.positions += disp

    # 2) Small random lattice strain
    #    Apply a random strain tensor to the cell vectors.
    #    strain[i,j] is the fractional change in cell vector i along direction j.
    strain = np.random.uniform(-max_strain, max_strain, size=(3, 3))
    #    New cell = (I + strain) @ old_cell  (row-vector convention)
    new_cell = (np.eye(3) + strain) @ atoms.cell.array
    atoms.cell = new_cell

    # --- INVARIANT 1: assert nat/ntyp consistency ---
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 6, f"Structure {i:03d}: expected 6 atoms, got {nat}"
    assert ntyp == 2, f"Structure {i:03d}: expected 2 species, got {ntyp}"

    fname = f'structures/structure_{i:03d}.cif'
    ase.io.write(fname, atoms)
    print(f"Wrote {fname}  (nat={nat}, ntyp={ntyp})")

print("\nDone: wrote 51 structures (000-050).")
