#!/usr/bin/env python3
"""
Generate 51 perturbed CsPbBr3 structures (structure_000.cif ... structure_050.cif).

structure_000.cif = unperturbed base structure (primitive cell, Pm-3m #221).
structure_001-050 = base + random atomic displacements (thermal motion) + small random lattice strain.

Temperature choice: T = 300 K
  CsPbBr3 cubic perovskite (Pm-3m) is the stable phase at room temperature.
  Unlike CsPbI3 (which is only stable above ~350 K in the cubic phase), CsPbBr3
  remains cubic down to ~230 K. 300 K is well within the stable range.

Displacement amplitudes (RMS, in Angstrom):
  We use literature-based Debye-Waller U values for CsPbBr3 at 300 K.
  The crystallographic U parameter is the mean-square displacement in A^2.
  The per-axis RMS displacement is sqrt(U) in A.

  Literature values for CsPbBr3 at 300 K (from X-ray/neutron diffraction studies):
    Br: U ~ 0.015 A^2  -> RMS = sqrt(0.015) ~ 0.122 A
    Pb: U ~ 0.008 A^2  -> RMS = sqrt(0.008) ~ 0.089 A
    Cs: U ~ 0.012 A^2  -> RMS = sqrt(0.012) ~ 0.110 A

  Physical justification:
    - Br is the lightest atom in the framework and has the largest thermal motion.
    - Pb is heavier (207 g/mol) and sits at the B-site with stronger bonding, so it moves less.
    - Cs is the A-site cation; it is large and weakly bound, giving it intermediate motion.

  Units: all amplitudes below are RMS displacements in Angstrom (A).
  The crystallographic U parameter (mean-square displacement) is U = RMS^2 in A^2.

Lattice strain: small random symmetric strain tensor, magnitude ~0.5% (0.005),
  representative of thermal expansion fluctuations.

No pymatgen is used. ASE only.
"""

import numpy as np
from ase.io import read, write
from ase import Atoms
import os

# --- Displacement amplitudes (RMS, in Angstrom) ---
# Derived from literature Debye-Waller U values for CsPbBr3 at 300 K:
#   U_Br = 0.015 A^2  -> RMS = sqrt(0.015) = 0.1225 A
#   U_Pb = 0.008 A^2  -> RMS = sqrt(0.008) = 0.0894 A
#   U_Cs = 0.012 A^2  -> RMS = sqrt(0.012) = 0.1095 A
RMS = {
    'Br': np.sqrt(0.015),  # 0.1225 A
    'Pb': np.sqrt(0.008),  # 0.0894 A
    'Cs': np.sqrt(0.012),  # 0.1095 A
}

print(f"Temperature: 300 K")
print(f"RMS displacements (Angstrom) and U values (A^2):")
for sp, rms in RMS.items():
    print(f"  {sp}: RMS = {rms:.4f} A  (U = {rms**2:.4f} A^2)")

# --- Read base structure ---
base = read('structures/base.cif')
print(f"\nBase structure: {base.get_chemical_formula()}")
print(f"  nat = {len(base)}, ntyp = {len(set(base.get_chemical_symbols()))}")
print(f"  cell = {base.cell[:].tolist()}")

# INVARIANT 1: nat/ntyp computed from actual Atoms object
nat = len(base)
typ = len(set(base.get_chemical_symbols()))
assert nat == 5, f"Expected 5 atoms in primitive CsPbBr3, got {nat}"
assert typ == 3, f"Expected 3 species, got {typ}"

# --- Generate structures ---
np.random.seed(42)  # reproducibility

for i in range(51):
    atoms = base.copy()
    
    if i > 0:
        # --- Random atomic displacements (thermal motion) ---
        # Each atom gets a Gaussian displacement with per-species RMS amplitude
        symbols = atoms.get_chemical_symbols()
        for idx, sym in enumerate(symbols):
            rms = RMS[sym]
            # Displacement in Cartesian Angstrom
            disp = np.random.normal(0.0, rms, size=3)
            atoms.positions[idx] += disp
        
        # --- Small random lattice strain ---
        # Symmetric strain tensor, magnitude ~0.5%
        strain_mag = 0.005
        # Random symmetric 3x3 matrix
        S = np.random.normal(0.0, strain_mag, size=(3, 3))
        S = 0.5 * (S + S.T)  # make symmetric
        # Apply strain to cell: new_cell = (I + S) @ old_cell
        old_cell = atoms.cell[:]
        new_cell = (np.eye(3) + S) @ old_cell
        atoms.set_cell(new_cell, scale_atoms=False)
    
    # INVARIANT 1: assert nat/ntyp for each structure
    assert len(atoms) == nat, f"Structure {i:03d}: nat changed from {nat} to {len(atoms)}"
    assert len(set(atoms.get_chemical_symbols())) == typ, \
        f"Structure {i:03d}: ntyp changed from {nat} to {len(set(atoms.get_chemical_symbols()))}"
    
    out_path = f'structures/structure_{i:03d}.cif'
    write(out_path, atoms)
    
    if i == 0:
        print(f"\nWrote {out_path} (unperturbed base)")
    elif i <= 3:
        print(f"Wrote {out_path} (perturbed)")

print(f"\nDone. Wrote 51 structures: structure_000.cif through structure_050.cif")
