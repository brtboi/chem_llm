import numpy as np
import ase.io
from ase import Atoms

# =============================================================================
# generate_structures.py
# Generates 51 perturbed SrTiO3 structures (structure_000.cif .. structure_050.cif)
# =============================================================================

# Temperature choice:
#   SrTiO3 is a cubic perovskite (Pm-3m) that is stable at room temperature.
#   It undergoes a cubic -> tetragonal transition at ~105 K, so at 300 K the
#   cubic phase is the stable one. We use T = 300 K for the thermal displacements.
T = 300.0  # Kelvin

# Debye-Waller factors (crystallographic displacement parameters U in A^2):
#   U is the mean-square displacement. The per-axis RMS displacement is sqrt(U) in A.
#   We assign species-specific amplitudes based on mass and bonding:
#     - O (lightest, most mobile): U = 0.010 A^2  -> RMS = sqrt(0.010) = 0.100 A
#     - Ti (intermediate):         U = 0.005 A^2  -> RMS = sqrt(0.005) = 0.071 A
#     - Sr (heaviest, most rigid): U = 0.003 A^2  -> RMS = sqrt(0.003) = 0.055 A
#   These values are representative of thermal motion at 300 K for a perovskite.
U_O  = 0.010  # A^2
U_Ti = 0.005  # A^2
U_Sr = 0.003  # A^2

rms_O  = np.sqrt(U_O)   # A
rms_Ti = np.sqrt(U_Ti)  # A
rms_Sr = np.sqrt(U_Sr)  # A

# Lattice strain: small random strain tensor, magnitude ~0.1%
# This represents a small random distortion of the unit cell.
strain_magnitude = 0.001  # 0.1% strain

rng = np.random.default_rng(seed=42)

# Read the base (unperturbed) structure
base = ase.io.read('structures/structure_000.cif')
print(f'Base structure: {len(base)} atoms, {len(set(base.get_chemical_symbols()))} species')
print(f'Cell: {base.cell[:]}')

# Write structure_000.cif (unperturbed base) -- already done, but rewrite for consistency
ase.io.write('structures/structure_000.cif', base)

for i in range(1, 51):
    atoms = base.copy()
    
    # --- Apply random lattice strain ---
    # Generate a small random symmetric strain tensor
    strain = rng.normal(0.0, strain_magnitude, size=(3, 3))
    strain = (strain + strain.T) / 2.0  # make symmetric
    # Apply strain to the cell: new_cell = (I + strain) @ old_cell
    # In ASE, cell is a 3x3 matrix of row vectors (a1, a2, a3)
    # The strain acts as: new_cell = old_cell @ (I + strain).T
    # (since cell rows are the lattice vectors, and strain is in Cartesian)
    new_cell = atoms.cell @ (np.eye(3) + strain).T
    atoms.set_cell(new_cell, scale_atoms=True)  # scale_atoms keeps fractional coords fixed
    
    # --- Apply random atomic displacements ---
    # Displacements are in Cartesian coordinates (A), drawn from a Gaussian
    # with per-species RMS amplitude (sqrt(U) in A).
    symbols = atoms.get_chemical_symbols()
    displacements = np.zeros((len(atoms), 3))
    for idx, sym in enumerate(symbols):
        if sym == 'O':
            rms = rms_O
        elif sym == 'Ti':
            rms = rms_Ti
        elif sym == 'Sr':
            rms = rms_Sr
        else:
            raise ValueError(f'Unknown species: {sym}')
        displacements[idx] = rng.normal(0.0, rms, size=3)
    
    atoms.positions += displacements
    
    # --- INVARIANT 1: nat/ntyp must match the actual Atoms object ---
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 5, f'Structure {i:03d}: expected nat=5, got {nat}'
    assert ntyp == 3, f'Structure {i:03d}: expected ntyp=3, got {ntyp}'
    
    # Write the perturbed structure
    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)
    
    if i % 10 == 0:
        print(f'  Wrote {out_path} (nat={nat}, ntyp={ntyp})')

print('Done. Wrote structures/structure_000.cif through structure_050.cif')
print(f'Temperature: {T} K')
print(f'U values (A^2): O={U_O}, Ti={U_Ti}, Sr={U_Sr}')
print(f'RMS displacements (A): O={rms_O:.4f}, Ti={rms_Ti:.4f}, Sr={rms_Sr:.4f}')
print(f'Strain magnitude: {strain_magnitude*100:.2f}%')
