import numpy as np
import ase.io
from ase import Atoms
import copy

# =============================================================================
# generate_structures.py
# Generate 51 perturbed PbTe structures (structure_000.cif ... structure_050.cif)
# =============================================================================

# Temperature choice:
# PbTe is stable in the rock-salt (NaCl, Fm-3m) phase from room temperature up
# to its melting point (~920 C). There is no phase transition below that, so
# T = 300 K is a physically meaningful choice for thermal displacements.
T = 300.0  # K

# Einstein-model RMS displacements (sqrt(U) in Angstrom):
#   sqrt(U) = sqrt(3 * kB * T / (m * omega^2))
# For PbTe at 300 K, typical Debye-Waller factors give:
#   Te (lighter, ~128 amu): RMS ~ 0.08 A  (larger amplitude, lighter atom)
#   Pb (heavier, ~207 amu): RMS ~ 0.06 A  (smaller amplitude, heavier atom)
# These are RMS values (sqrt(U)) in Angstrom, NOT U itself.
# U_Te = 0.08^2 = 0.0064 A^2, U_Pb = 0.06^2 = 0.0036 A^2
RMS_Te = 0.08  # Angstrom (per-axis RMS displacement)
RMS_Pb = 0.06  # Angstrom (per-axis RMS displacement)

# Small random lattice strain: +/- 1% per axis (typical for thermal expansion
# fluctuations at 300 K in PbTe, which has a thermal expansion coefficient
# of ~30e-6 /K, so over a 10 K fluctuation ~0.03%, but we use 1% to be
# representative of structural noise in a DFT perturbation study).
STRAIN_MAX = 0.01  # 1% max strain per axis

np.random.seed(42)

# Read the base (unperturbed) structure
base = ase.io.read('structures/structure_000.cif')
print(f"Base structure: {len(base)} atoms, symbols: {base.get_chemical_symbols()}")
print(f"Base cell: {base.cell[:].tolist()}")

# INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
nat_base = len(base)
typ_base = len(set(base.get_chemical_symbols()))
print(f"Base nat={nat_base}, ntyp={typ_base}")
assert nat_base == 2, f"Expected 2 atoms in base PbTe, got {nat_base}"
assert typ_base == 2, f"Expected 2 species in base PbTe, got {typ_base}"

for i in range(51):
    if i == 0:
        # structure_000: unperturbed base
        atoms = copy.deepcopy(base)
    else:
        atoms = copy.deepcopy(base)
        
        # --- Random atomic displacements (thermal motion) ---
        # Displace each atom in Cartesian coordinates by a Gaussian random
        # vector with per-species standard deviation = RMS value.
        # Displacements are in Angstrom (Cartesian).
        symbols = atoms.get_chemical_symbols()
        positions = atoms.positions.copy()  # Angstrom
        for j, sym in enumerate(symbols):
            if sym == 'Te':
                sigma = RMS_Te
            elif sym == 'Pb':
                sigma = RMS_Pb
            else:
                raise ValueError(f"Unexpected species {sym}")
            # Gaussian displacement in each Cartesian direction (Angstrom)
            positions[j] += np.random.normal(0.0, sigma, size=3)
        atoms.positions = positions
        
        # --- Small random lattice strain ---
        # Apply a random diagonal strain tensor (no shear) to the cell.
        # strain_ij is dimensionless; new_cell = old_cell * (I + strain)
        strain = np.random.uniform(-STRAIN_MAX, STRAIN_MAX, size=3)
        # Apply strain to the cell vectors (Angstrom)
        old_cell = atoms.cell[:].copy()
        new_cell = old_cell * (1.0 + strain)  # element-wise per axis
        atoms.set_cell(new_cell, scale_atoms=True)  # scale positions with cell
    
    # INVARIANT 1: assert nat/ntyp for each structure
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == nat_base, f"Structure {i:03d}: nat={nat}, expected {nat_base}"
    assert ntyp == typ_base, f"Structure {i:03d}: ntyp={ntyp}, expected {typ_base}"
    
    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)
    
    if i < 3 or i == 50:
        print(f"Wrote {out_path}: nat={nat}, ntyp={ntyp}")

print(f"\nDone. Wrote 51 structures: structure_000.cif through structure_050.cif")
print(f"Temperature: {T} K")
print(f"RMS displacements: Te={RMS_Te} A, Pb={RMS_Pb} A (per-axis, sqrt(U) in Angstrom)")
print(f"Max strain per axis: {STRAIN_MAX*100:.1f}%")
