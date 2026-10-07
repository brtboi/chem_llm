import numpy as np
import ase.io
from ase import Atoms
from ase.spacegroup import get_spacegroup

# =============================================================================
# generate_structures.py
# Generates 51 perturbed CsPbBr3 structures (structure_000 to structure_050)
# from the primitive cell saved in structures/base_primitive.cif.
# =============================================================================

# --- Temperature choice ---
# CsPbBr3 cubic perovskite (Pm-3m) is stable at room temperature and above.
# The cubic phase is the high-temperature phase; it is stable from ~300 K up
# to the melting point (~1000 K). We choose T = 300 K (room temperature),
# which is well within the cubic phase stability range.
T = 300.0  # Kelvin

# --- Displacement amplitudes ---
# We use the Einstein model to estimate mean-square displacements:
#   <u^2> = 3 k_B T / (M omega^2)
# For a rough estimate, we use a characteristic phonon frequency for each species.
# In perovskites, the light halide (Br) has higher-frequency optical modes,
# while the heavy A-site cation (Cs) has lower-frequency modes.
#
# Typical Debye temperatures / characteristic frequencies:
#   Cs:  ~100 K  -> omega ~ 2*pi*100*1.38e-23/1.05e-34 ~ 8.3e12 rad/s
#   Pb:  ~100 K  -> omega ~ 8.3e12 rad/s
#   Br:  ~200 K  -> omega ~ 1.66e13 rad/s
#
# Using <u^2> = 3 k_B T / (M omega^2):
#   k_B = 1.38e-23 J/K
#   M(Cs) = 133 amu = 133 * 1.66e-27 kg
#   M(Pb) = 207 amu = 207 * 1.66e-27 kg
#   M(Br) =  80 amu =  80 * 1.66e-27 kg
#
# For Cs: <u^2> = 3 * 1.38e-23 * 300 / (133*1.66e-27 * (8.3e12)^2)
#          = 1.242e-20 / (1.445e-23 * 6.89e25)
#          = 1.242e-20 / 9.96e2
#          = 1.25e-23 m^2 = 0.00125 A^2
#   RMS = sqrt(0.00125) = 0.035 A
#
# For Pb: <u^2> = 3 * 1.38e-23 * 300 / (207*1.66e-27 * (8.3e12)^2)
#          = 1.242e-20 / (3.436e-25 * 6.89e25)
#          = 1.242e-20 / 23.67
#          = 5.25e-22 m^2 = 0.00525 A^2
#   Wait, that gives Pb MORE displacement than Cs, which is wrong.
#   Let me reconsider: heavier atoms have LOWER frequencies, so <u^2> is LARGER.
#   Actually, for the same frequency, heavier atoms have smaller <u^2>.
#   But in real perovskites, the A-site cation (Cs) is in a soft potential well
#   and has larger displacements than the B-site (Pb).
#
# Let's use a simpler heuristic based on known B-factor values from XRD/neutron data:
#   Cs in CsPbBr3: B ~ 2.0-3.0 A^2 (U = B/(8*pi^2) ~ 0.025-0.038 A^2)
#   Pb in CsPbBr3: B ~ 1.0-2.0 A^2 (U ~ 0.013-0.025 A^2)
#   Br in CsPbBr3: B ~ 1.0-1.5 A^2 (U ~ 0.013-0.019 A^2)
#
# We use U (mean-square displacement in A^2) values:
#   Cs: U = 0.030 A^2  -> RMS = sqrt(0.030) = 0.173 A
#   Pb: U = 0.020 A^2  -> RMS = sqrt(0.020) = 0.141 A
#   Br: U = 0.015 A^2  -> RMS = sqrt(0.015) = 0.122 A
#
# These are representative of thermal motion at 300 K for this phase.
# The per-axis RMS displacement is sqrt(U) in Angstrom.

U_Cs = 0.030  # A^2, mean-square displacement for Cs
U_Pb = 0.020  # A^2, mean-square displacement for Pb
U_Br = 0.015  # A^2, mean-square displacement for Br

# Per-axis RMS displacement (in Angstrom):
RMS_Cs = np.sqrt(U_Cs)  # ~0.173 A
RMS_Pb = np.sqrt(U_Pb)  # ~0.141 A
RMS_Br = np.sqrt(U_Br)  # ~0.122 A

# --- Lattice strain ---
# Small random lattice strain to represent thermal expansion/contraction.
# Typical thermal strain in perovskites at 300 K is ~0.1-0.5%.
# We use a maximum strain of 0.002 (0.2%) per axis.
max_strain = 0.002  # dimensionless, 0.2%

# --- Read base structure ---
base = ase.io.read('structures/base_primitive.cif')
print(f"Base structure: {len(base)} atoms, cell = {base.cell.array}")

# Verify base structure
assert len(base) == 5, f"Expected 5 atoms in primitive cell, got {len(base)}"
assert set(base.get_chemical_symbols()) == {'Cs', 'Pb', 'Br'}, \
    f"Unexpected elements: {set(base.get_chemical_symbols())}"

# --- Generate 51 structures ---
np.random.seed(42)  # Reproducibility

for i in range(51):
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()
        
        # Apply random lattice strain
        # strain is a 3x3 matrix added to the identity
        strain = np.random.uniform(-max_strain, max_strain, size=(3, 3))
        # Make strain symmetric (physical strain tensor is symmetric)
        strain = 0.5 * (strain + strain.T)
        # New cell = (I + strain) @ old_cell
        new_cell = (np.eye(3) + strain) @ base.cell.array
        atoms.cell = new_cell
        
        # Apply random atomic displacements
        # Displacements are in Cartesian coordinates (Angstrom)
        # Each atom gets a Gaussian displacement with per-species RMS
        for j, atom in enumerate(atoms):
            symbol = atom.symbol
            if symbol == 'Cs':
                rms = RMS_Cs
            elif symbol == 'Pb':
                rms = RMS_Pb
            elif symbol == 'Br':
                rms = RMS_Br
            else:
                raise ValueError(f"Unknown element: {symbol}")
            # Gaussian displacement in each Cartesian direction
            disp = np.random.normal(0, rms, size=3)
            atoms.positions[j] += disp
        
        # Wrap positions back into the cell (fractional coordinates in [0, 1))
        atoms.wrap()
    
    # INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 5, f"Structure {i:03d}: expected 5 atoms, got {nat}"
    assert ntyp == 3, f"Structure {i:03d}: expected 3 types, got {ntyp}"
    
    # Write structure
    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)
    
    if i % 10 == 0:
        print(f"  Written {out_path} (nat={nat}, ntyp={ntyp})")

print(f"\nDone. Generated 51 structures in structures/.")
print(f"Temperature: {T} K")
print(f"Displacement amplitudes (U in A^2, RMS in A):")
print(f"  Cs: U={U_Cs} A^2, RMS={RMS_Cs:.4f} A")
print(f"  Pb: U={U_Pb} A^2, RMS={RMS_Pb:.4f} A")
print(f"  Br: U={U_Br} A^2, RMS={RMS_Br:.4f} A")
print(f"Max lattice strain: {max_strain} ({max_strain*100:.1f}%)")
