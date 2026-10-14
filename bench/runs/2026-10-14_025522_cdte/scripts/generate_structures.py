import numpy as np
import ase.io

# Read the base (unperturbed) structure
base = ase.io.read('structures/structure_000.cif')

# Temperature choice: CdTe zincblende is stable at room temperature (melting point ~958 C).
# 300 K is appropriate for thermal displacements.
T = 300  # K

# Crystallographic displacement parameters U (mean-square displacements in A^2):
# Cd (Z=48, 112 amu) is lighter than Te (Z=52, 128 amu), so Cd moves more.
# U_Cd = 0.005 A^2, U_Te = 0.003 A^2
# Per-axis RMS displacement = sqrt(U): Cd ~0.071 A, Te ~0.055 A
U_Cd = 0.005  # A^2
U_Te = 0.003  # A^2
rms_Cd = np.sqrt(U_Cd)  # A
rms_Te = np.sqrt(U_Te)  # A

# Random lattice strain: ~0.5% (typical for thermal expansion / small perturbations)
strain_magnitude = 0.005

rng = np.random.default_rng(42)  # fixed seed for reproducibility

for i in range(51):
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()
        # Apply random atomic displacements (thermal motion)
        disp = np.zeros((len(atoms), 3))
        for j, sym in enumerate(atoms.get_chemical_symbols()):
            if sym == 'Cd':
                disp[j] = rng.normal(0, rms_Cd, 3)
            elif sym == 'Te':
                disp[j] = rng.normal(0, rms_Te, 3)
        atoms.positions += disp
        # Apply small random lattice strain
        # Strain tensor: symmetric 3x3 matrix, applied to the cell vectors
        strain = rng.uniform(-strain_magnitude, strain_magnitude, (3, 3))
        strain = (strain + strain.T) / 2.0  # make symmetric
        # New cell = old cell * (I + strain)
        # In ASE, cell is a 3x3 matrix of row vectors (a, b, c)
        # To apply strain to the cell: new_cell = (I + strain).T @ old_cell
        # This is equivalent to transforming the cell vectors by the strain tensor
        new_cell = (np.eye(3) + strain).T @ np.array(atoms.cell)
        atoms.set_cell(new_cell, scale_atoms=True)
    
    # INVARIANT 1: nat and ntyp must match the actual structure
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Expected 2 atoms, got {nat}"
    assert ntyp == 2, f"Expected 2 species, got {ntyp}"
    
    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)

print(f"Generated 51 structures: structure_000.cif (unperturbed) through structure_050.cif")
print(f"Temperature: {T} K")
print(f"U_Cd = {U_Cd} A^2 (RMS = {rms_Cd:.4f} A), U_Te = {U_Te} A^2 (RMS = {rms_Te:.4f} A)")
print(f"Strain magnitude: {strain_magnitude*100:.1f}%")
