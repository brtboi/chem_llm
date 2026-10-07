import numpy as np
from ase.io import read, write
from ase import Atoms

# =============================================================================
# generate_structures.py
# Reads the primitive-cell GaAs CIF and writes 51 structures:
#   structure_000.cif : unperturbed base
#   structure_001-050.cif : random thermal displacements + small random strain
# =============================================================================

# --- Temperature choice ---
# Zincblende GaAs is the equilibrium phase at room temperature (stable up to
# ~1200 C). We use T = 300 K, a temperature at which this phase is definitely
# stable and thermal displacements are physically meaningful.
T = 300.0  # Kelvin

# --- Einstein-model mean-square displacements ---
# For a 3D Einstein oscillator, the mean-square displacement is:
#   <u^2> = (3*hbar / (2*m*omega)) * coth(hbar*omega / (2*k_B*T))
# where omega = k_B*theta_E / hbar, so:
#   <u^2> = (3*hbar^2 / (2*m*k_B*theta_E)) * coth(theta_E / (2*T))
#
# The per-axis RMS displacement is sqrt(<u^2>/3) = sqrt(U) in Angstrom.
# U (the crystallographic displacement parameter) is the mean-square
# displacement in A^2; sqrt(U) is the per-axis RMS in A.
#
# Einstein theta values (characteristic vibrational temperatures):
#   Ga: theta_E = 300 K  (lighter atom, lower frequency -> larger displacement)
#   As: theta_E = 350 K  (heavier atom, slightly stiffer -> smaller displacement)
# These are representative values for GaAs; the exact values depend on the
# phonon density of states, but the Einstein model captures the key physics:
# lighter atoms move more, and displacements decrease with increasing T.

hbar = 1.054571817e-34      # J*s
k_B = 1.380649e-23          # J/K
u_to_kg = 1.66053906660e-27 # kg per atomic mass unit

m_Ga = 69.723 * u_to_kg     # kg
m_As = 74.922 * u_to_kg     # kg

theta_Ga = 300.0  # K
theta_As = 350.0  # K

def einstein_rms(m, theta, T):
    """Per-axis RMS displacement in Angstrom for an Einstein oscillator.
    
    <u^2> = (3*hbar^2 / (2*m*k_B*theta)) * coth(theta / (2*T))
    Per-axis RMS = sqrt(<u^2>/3) in m, converted to Angstrom.
    """
    # <u^2> in m^2
    u2 = (3.0 * hbar**2 / (2.0 * m * k_B * theta)) \
         * (1.0 / np.tanh(theta / (2.0 * T)))
    # per-axis RMS in m, convert to Angstrom
    rms_m = np.sqrt(u2 / 3.0)
    return rms_m * 1e10  # m -> Angstrom

rms_Ga = einstein_rms(m_Ga, theta_Ga, T)  # A
rms_As = einstein_rms(m_As, theta_As, T)  # A

print(f"T = {T} K")
print(f"Ga: theta_E = {theta_Ga} K, per-axis RMS displacement = {rms_Ga:.4f} A")
print(f"As: theta_E = {theta_As} K, per-axis RMS displacement = {rms_As:.4f} A")

# Sanity check: RMS displacements should be in the range 0.01-0.5 A for
# typical semiconductors at 300 K
assert 0.01 < rms_Ga < 0.5, f"Ga RMS {rms_Ga} A out of expected range"
assert 0.01 < rms_As < 0.5, f"As RMS {rms_As} A out of expected range"

# --- Random lattice strain ---
# Small random strain tensor with magnitude ~0.5% (typical of thermal or
# defect-induced strain in a crystal). Applied as a uniform strain to the cell.
strain_magnitude = 0.005  # 0.5%

# --- Read base structure ---
base = read('structures/GaAs_primitive.cif')
print(f"Base structure: {len(base)} atoms, symbols = {base.get_chemical_symbols()}")

# INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
nat = len(base)
ntyp = len(set(base.get_chemical_symbols()))
print(f"nat = {nat}, ntyp = {ntyp}")

# --- Write structure_000.cif (unperturbed) ---
write('structures/structure_000.cif', base)
print("Wrote structures/structure_000.cif")

# --- Write perturbed structures 001-050 ---
np.random.seed(42)  # reproducibility

for i in range(1, 51):
    atoms = base.copy()
    
    # Random atomic displacements (thermal motion)
    # Each atom gets a random displacement drawn from a Gaussian with
    # per-species RMS amplitude (per-axis, in Angstrom).
    for atom in atoms:
        symbol = atom.symbol
        if symbol == 'Ga':
            rms = rms_Ga
        elif symbol == 'As':
            rms = rms_As
        else:
            raise ValueError(f"Unknown element {symbol}")
        # Displacement in Cartesian coordinates (Angstrom)
        disp = np.random.normal(0.0, rms, size=3)
        atom.position += disp
    
    # Small random lattice strain
    # Generate a random symmetric strain tensor with magnitude ~0.5%
    strain = np.random.normal(0.0, strain_magnitude / np.sqrt(6), size=(3, 3))
    strain = 0.5 * (strain + strain.T)  # make symmetric
    # Apply strain to the cell: new_cell = (I + strain) @ old_cell
    new_cell = (np.eye(3) + strain) @ atoms.cell[:]
    atoms.cell = new_cell
    
    # INVARIANT 1: assert nat/ntyp from the actual Atoms object
    assert len(atoms) == nat, f"Structure {i:03d}: nat changed from {nat} to {len(atoms)}"
    assert len(set(atoms.get_chemical_symbols())) == ntyp, \
        f"Structure {i:03d}: ntyp changed from {ntyp} to {len(set(atoms.get_chemical_symbols()))}"
    
    fname = f'structures/structure_{i:03d}.cif'
    write(fname, atoms)
    print(f"Wrote {fname}")

print("Done: 51 structures written (000-050).")
