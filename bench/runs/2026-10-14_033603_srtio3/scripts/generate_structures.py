import numpy as np
import ase.io
from ase import Atoms

# =============================================================================
# generate_structures.py
# Generates 51 perturbed SrTiO3 structures (structure_000.cif - structure_050.cif)
# =============================================================================

# --- Temperature choice ---
# SrTiO3 is a cubic perovskite (Pm-3m) that is stable at room temperature and above.
# The cubic phase is the high-temperature phase; below ~105 K it undergoes a
# ferroelectric distortion to tetragonal (P4mm), and below ~260 K it becomes
# rhombohedral (R3m). So the cubic phase is stable at T >= ~105 K.
# We choose T = 300 K (room temperature), well within the cubic stability range.
T = 300.0  # Kelvin

# --- Displacement amplitudes (RMS, in Angstrom) ---
# For a harmonic oscillator at temperature T, the mean-square displacement is:
#   <u^2> = 3 k_B T / (M omega^2)
# where omega is the characteristic vibrational frequency.
# In a perovskite, the light O atoms have the highest frequencies and thus the
# smallest displacements; the heavier Sr and Ti have lower frequencies and larger
# displacements. However, the bond stiffness also matters: Ti-O bonds are strong,
# so Ti is relatively stiff. Sr is the heaviest and most weakly bound (it sits in
# the 12-fold coordination site with relatively soft modes), so Sr moves the most.
#
# We use Debye-Waller-type estimates. For SrTiO3 at 300 K, typical RMS
# displacements from neutron diffraction / MD are approximately:
#   Sr: ~0.08 A  (heaviest, softest modes)
#   Ti: ~0.05 A  (intermediate mass, strong Ti-O bonds)
#   O:  ~0.04 A  (lightest, highest frequency modes)
#
# These are RMS displacements (sqrt(U) in A), where U is the mean-square
# displacement parameter in A^2. We use them directly as the standard deviation
# of a Gaussian displacement distribution per Cartesian axis.
RMS = {
    'Sr': 0.08,  # A, RMS displacement per axis
    'Ti': 0.05,  # A, RMS displacement per axis
    'O':  0.04,  # A, RMS displacement per axis
}

# --- Lattice strain ---
# Small random homogeneous strain, representative of thermal expansion fluctuations.
# Typical thermal strain in SrTiO3 over a 10 K range is ~1e-4.
# We use a standard deviation of 0.001 (0.1%) per strain component, which is
# a reasonable upper bound for local strain fluctuations at 300 K.
STRAIN_STD = 0.001  # dimensionless, standard deviation of each strain component

# --- Reproducibility ---
np.random.seed(42)

# --- Read base structure ---
base = ase.io.read('structures/SrTiO3_base.cif')
print(f"Base structure: {len(base)} atoms, cell = {base.cell[:].tolist()}")

# --- Write structure_000 (unperturbed) ---
ase.io.write('structures/structure_000.cif', base)
print("Wrote structures/structure_000.cif (unperturbed)")

# --- Generate perturbed structures ---
for n in range(1, 51):
    atoms = base.copy()
    
    # Apply random lattice strain
    # strain tensor: symmetric 3x3, each component drawn from N(0, STRAIN_STD)
    strain = np.random.normal(0.0, STRAIN_STD, size=(3, 3))
    strain = 0.5 * (strain + strain.T)  # make symmetric
    # New cell = (I + strain) @ old_cell  (strain applied in Cartesian)
    old_cell = np.array(atoms.cell[:])
    new_cell = (np.eye(3) + strain) @ old_cell
    atoms.set_cell(new_cell, scale_atoms=True)  # scale atoms to stay in cell
    
    # Apply random atomic displacements (thermal motion)
    # Each atom gets a Gaussian displacement with species-specific RMS
    disp = np.zeros((len(atoms), 3))
    for i, sym in enumerate(atoms.get_chemical_symbols()):
        rms = RMS[sym]
        disp[i] = np.random.normal(0.0, rms, size=3)
    atoms.positions += disp
    
    # Wrap positions into the unit cell
    atoms.wrap()
    
    # --- INVARIANT 1: nat and ntyp must match the actual Atoms object ---
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 5, f"Structure {n:03d}: expected nat=5, got {nat}"
    assert ntyp == 3, f"Structure {n:03d}: expected ntyp=3, got {ntyp}"
    
    out_path = f'structures/structure_{n:03d}.cif'
    ase.io.write(out_path, atoms)
    
    if n <= 3 or n % 10 == 0:
        print(f"Wrote {out_path} (nat={nat}, ntyp={ntyp})")

print("\nDone. Generated 51 structures (structure_000.cif - structure_050.cif)")
print(f"Temperature: {T} K")
print(f"RMS displacements (A): Sr={RMS['Sr']}, Ti={RMS['Ti']}, O={RMS['O']}")
print(f"Strain std: {STRAIN_STD}")
