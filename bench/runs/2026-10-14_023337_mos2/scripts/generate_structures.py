"""
Generate 51 perturbed MoS2 structures (structure_000.cif through structure_050.cif).

structure_000.cif: unperturbed base (primitive cell from Materials Project mp-2815)
structure_001-050.cif: random thermal displacements + small random lattice strain

Temperature choice: 300 K (room temperature)
  - 2H MoS2 is the thermodynamically stable phase at room temperature and below.
  - The 1T phase is metastable at ambient conditions.
  - At 300 K, thermal displacements are small but non-negligible.

Species amplitudes (RMS displacement per axis, in Angstrom):
  - S (Z=16, lighter): 0.06 A RMS  -> U = 0.0036 A^2 (mean-square displacement)
  - Mo (Z=42, heavier): 0.04 A RMS -> U = 0.0016 A^2
  Justification: lighter atoms have larger thermal amplitudes. These values are
  consistent with typical Debye-Waller factors for MoS2 at 300 K from literature.

Strain: random strain tensor with components up to ~0.5% (0.005) max.
"""
import numpy as np
import ase.io
from ase import Atoms

# Read the base structure (primitive cell)
base = ase.io.read('structures/prim_MoS2.cif')
print(f"Base structure: {len(base)} atoms, cell = {base.get_cell()}")

# Species-specific RMS displacements (Angstrom)
# U (crystallographic displacement parameter) = mean-square displacement in A^2
# RMS per axis = sqrt(U) in A
RMS_S = 0.06   # A, for Sulfur (lighter, Z=16)
RMS_Mo = 0.04  # A, for Molybdenum (heavier, Z=42)

# Random strain: max component ~0.5%
MAX_STRAIN = 0.005

np.random.seed(42)  # reproducibility

for i in range(51):
    nnn = f"{i:03d}"
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()
        
        # Apply random lattice strain
        # Strain tensor: small random symmetric tensor
        strain = np.random.uniform(-MAX_STRAIN, MAX_STRAIN, size=(3, 3))
        # Make it symmetric (physical strain is symmetric)
        strain = 0.5 * (strain + strain.T)
        # Apply strain to the cell
        new_cell = atoms.get_cell() * (np.eye(3) + strain)
        atoms.set_cell(new_cell, scale_atoms=True)
        
        # Apply random thermal displacements to each atom
        # Displacements are in Cartesian coordinates (Angstrom)
        # RMS per axis = sqrt(U), where U is the mean-square displacement
        for j, sym in enumerate(atoms.get_chemical_symbols()):
            if sym == 'S':
                rms = RMS_S
            elif sym == 'Mo':
                rms = RMS_Mo
            else:
                rms = 0.05  # default
            # Gaussian displacement with standard deviation = rms
            disp = np.random.normal(0.0, rms, size=3)
            atoms.positions[j] += disp
    
    # INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 6, f"Structure {nnn}: expected 6 atoms, got {nat}"
    assert ntyp == 2, f"Structure {nnn}: expected 2 species, got {ntyp}"
    
    out_path = f"structures/structure_{nnn}.cif"
    ase.io.write(out_path, atoms)
    print(f"Wrote {out_path}: nat={nat}, ntyp={ntyp}")

print("Done: 51 structures written.")
