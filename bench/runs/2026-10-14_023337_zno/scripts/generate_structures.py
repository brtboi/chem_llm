import numpy as np
import ase.io
from ase import Atoms

# =============================================================================
# generate_structures.py
# Generates 51 wurtzite ZnO structures: structure_000.cif (unperturbed base)
# and structure_001.cif ... structure_050.cif (perturbed copies).
#
# Perturbations:
#   1. Random atomic displacements representative of thermal motion at 300 K.
#   2. Small random lattice strain (~1% per component).
#
# Temperature justification:
#   Wurtzite ZnO is the thermodynamically stable phase at ambient conditions
#   (room temperature and pressure). It remains stable up to ~1000 K before
#   transforming to the rock-salt (NaCl) phase under pressure. 300 K is
#   therefore an appropriate temperature for thermal displacements.
#
# Displacement amplitudes (per-axis RMS, in Angstrom):
#   O:  0.08 A  (lighter atom, 16 amu, moves more)
#   Zn: 0.05 A  (heavier atom, 65 amu, moves less)
#   These are typical thermal displacement amplitudes for ZnO at 300 K.
#   U (crystallographic displacement parameter) = mean-square displacement in A^2.
#   Per-axis RMS displacement = sqrt(U) in A.
#   So U_O = (0.08)^2 = 0.0064 A^2, U_Zn = (0.05)^2 = 0.0025 A^2.
#
# Random strain: ~1% per component, applied as a small random deformation
#   to the lattice vectors.
# =============================================================================

np.random.seed(42)  # reproducibility

# Read the primitive base structure
base = ase.io.read('structures/zno_primitive.cif')
print(f"Base structure: {len(base)} atoms, species: {base.get_chemical_symbols()}")

# Per-species RMS displacement amplitudes (in Angstrom)
# O is lighter (16 amu) than Zn (65 amu), so O moves more
rms_disp = {'O': 0.08, 'Zn': 0.05}

# Small random strain: ~1% per component
strain_max = 0.01

for i in range(51):
    if i == 0:
        # Unperturbed base structure
        atoms = base.copy()
    else:
        atoms = base.copy()
        
        # Apply small random lattice strain
        # strain tensor: symmetric, small random values
        strain = np.random.uniform(-strain_max, strain_max, (3, 3))
        # Make it symmetric (physical strain is symmetric)
        strain = 0.5 * (strain + strain.T)
        # Apply to cell: new_cell = (I + strain) @ old_cell
        # In ASE, cell is a 3x3 matrix of lattice vectors
        old_cell = np.array(atoms.cell)
        new_cell = (np.eye(3) + strain) @ old_cell
        atoms.cell = new_cell
        # Keep positions in fractional coordinates (they stay the same)
        # but update Cartesian positions
        atoms.positions = atoms.get_scaled_positions() @ new_cell
        
        # Apply random atomic displacements (thermal motion)
        # Displacements are in Cartesian coordinates (Angstrom)
        for j, sym in enumerate(atoms.get_chemical_symbols()):
            amp = rms_disp[sym]
            # Random displacement: Gaussian, per-axis RMS = amp
            disp = np.random.normal(0.0, amp, 3)
            atoms.positions[j] += disp
    
    # INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 4, f"Structure {i:03d}: expected 4 atoms, got {nat}"
    assert ntyp == 2, f"Structure {i:03d}: expected 2 species, got {ntyp}"
    
    # Write the structure
    out_path = f'structures/structure_{i:03d}.cif'
    ase.io.write(out_path, atoms)
    
    if i == 0:
        print(f"  {out_path}: unperturbed base ({nat} atoms, {ntyp} species)")
    else:
        print(f"  {out_path}: perturbed ({nat} atoms, {ntyp} species)")

print(f"\nGenerated 51 structures in structures/")
print("Done.")
