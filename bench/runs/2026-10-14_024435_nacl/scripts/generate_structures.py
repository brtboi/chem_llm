import numpy as np
import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

# =============================================================================
# generate_structures.py
# Generates 51 NaCl structures (000-050) with thermal displacements + strain.
# =============================================================================

# Temperature choice: 300 K
# NaCl (rock salt) is stable at room temperature and well above it (melting
# point ~1074 K). 300 K is a representative ambient temperature at which this
# phase is the stable ground state. No high-temperature phase transition is
# relevant here.
T = 300.0  # K

# Species-specific displacement amplitudes (RMS, in Angstrom)
# Derived from Debye-Waller factors: U (mean-square displacement, A^2) for
# NaCl at 300 K from crystallographic data:
#   Na: U ~ 0.0055 A^2  -> RMS = sqrt(U) ~ 0.074 A
#   Cl: U ~ 0.0045 A^2  -> RMS = sqrt(U) ~ 0.067 A
# Na is lighter (23 amu vs 35.5 amu) and more weakly bound in the rock-salt
# lattice, so it has a larger amplitude. These values are representative of
# thermal motion at 300 K.
RMS_Na = 0.074  # A (RMS displacement per axis for Na)
RMS_Cl = 0.067  # A (RMS displacement per axis for Cl)

# Lattice strain: small random strain tensor, magnitude ~0.1%
# This represents slight deviations from the ideal cubic lattice due to
# thermal expansion fluctuations or sample inhomogeneity.
STRAIN_MAG = 0.001  # dimensionless (0.1%)

# Random seed for reproducibility
np.random.seed(42)

# Read the base (unperturbed) structure
base = ase.io.read('structures/structure_000_base.cif')
print(f"Base structure: nat={len(base)}, ntyp={len(set(base.get_chemical_symbols()))}")

# Write structure_000.cif (unperturbed)
ase.io.write('structures/structure_000.cif', base)
print("Wrote structures/structure_000.cif (unperturbed)")

for n in range(1, 51):
    atoms = base.copy()
    
    # --- Random atomic displacements (thermal motion) ---
    # Each atom gets a Gaussian displacement with species-specific RMS amplitude.
    # Displacements are in Cartesian coordinates (Angstrom).
    symbols = atoms.get_chemical_symbols()
    for i, sym in enumerate(symbols):
        if sym == 'Na':
            rms = RMS_Na
        elif sym == 'Cl':
            rms = RMS_Cl
        else:
            raise ValueError(f"Unexpected element {sym}")
        disp = np.random.normal(0.0, rms, size=3)
        atoms.positions[i] += disp
    
    # --- Small random lattice strain ---
    # Apply a small random strain tensor to the cell.
    # strain = [[e11, e12, e13], [e12, e22, e23], [e13, e23, e33]]
    # with each component drawn from N(0, STRAIN_MAG/3) so the overall
    # strain magnitude is ~STRAIN_MAG.
    strain = np.random.normal(0.0, STRAIN_MAG / np.sqrt(3), size=(3, 3))
    # Symmetrize the strain tensor
    strain = 0.5 * (strain + strain.T)
    # New cell = (I + strain) @ old_cell
    old_cell = np.array(atoms.cell)
    new_cell = (np.eye(3) + strain) @ old_cell
    atoms.set_cell(new_cell, scale_atoms=True)
    
    # --- INVARIANT 1: nat and ntyp must match the actual Atoms object ---
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Structure {n:03d}: expected nat=2, got {nat}"
    assert ntyp == 2, f"Structure {n:03d}: expected ntyp=2, got {ntyp}"
    
    out_path = f'structures/structure_{n:03d}.cif'
    ase.io.write(out_path, atoms)
    print(f"Wrote {out_path} (nat={nat}, ntyp={ntyp})")

print("\nDone: 51 structures written (000-050).")
