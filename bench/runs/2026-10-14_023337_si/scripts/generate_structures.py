import numpy as np
import ase.io
import os

# =============================================================================
# generate_structures.py
# Generates 51 perturbed Si structures (structure_000.cif through structure_050.cif)
# 000 = unperturbed base, 001-050 = thermal displacements + random strain
# =============================================================================

# Temperature choice: T = 300 K
# Diamond-structure Si is the stable phase at room temperature (below ~1200 K).
# 300 K is well within the stability range, so thermal displacements at this
# temperature are physically meaningful.
T = 300.0  # K

# Per-species amplitude (RMS displacement in Angstroms):
# Only Si in this structure. Lighter atoms move more, but here we have only Si.
# 0.02 A is a reasonable RMS displacement for Si at 300 K.
# This is consistent with experimental Debye-Waller factors for Si at 300 K.
# U = (0.02 A)^2 = 4e-4 A^2 (mean-square displacement)
# RMS per axis = 0.02 A
species_amplitudes = {
    'Si': 0.02  # RMS displacement in Angstroms (sqrt(U) where U = 4e-4 A^2)
}

# Random strain: small random lattice strain, ~0.1% (0.001)
# This represents small deviations from the ideal lattice constant.
max_strain = 0.001  # 0.1% strain

# Read the base structure (already reduced to primitive cell)
base = ase.io.read('structures/si_primitive.cif')
print(f"Base structure: {len(base)} atoms, cell = {base.cell[:].tolist()}")

# Create output directory
os.makedirs('structures', exist_ok=True)

# Write structure_000.cif (unperturbed base)
ase.io.write('structures/structure_000.cif', base)
print("Wrote structures/structure_000.cif (unperturbed)")

# Generate perturbed structures 001-050
np.random.seed(42)  # Reproducibility

for i in range(1, 51):
    atoms = base.copy()
    
    # Apply random thermal displacements
    # Each atom gets a random displacement drawn from a Gaussian distribution
    # with standard deviation = species_amplitudes[species]
    positions = atoms.get_positions().copy()
    for j, atom in enumerate(atoms):
        species = atom.symbol
        amp = species_amplitudes[species]
        # Random displacement in Cartesian coordinates (Angstroms)
        disp = np.random.normal(0.0, amp, size=3)
        positions[j] += disp
    atoms.set_positions(positions)
    
    # Apply small random lattice strain
    # Random strain tensor: symmetric, traceless (volume-preserving) + small isotropic part
    # For simplicity, apply a small random strain to each cell vector independently
    cell = atoms.cell[:].copy()
    for axis in range(3):
        strain = np.random.uniform(-max_strain, max_strain)
        cell[axis] *= (1.0 + strain)
    atoms.set_cell(cell, scale_atoms=False)
    
    # INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Structure {i:03d}: expected 2 atoms, got {nat}"
    assert ntyp == 1, f"Structure {i:03d}: expected 1 species, got {ntyp}"
    
    # Write the structure
    fname = f'structures/structure_{i:03d}.cif'
    ase.io.write(fname, atoms)
    print(f"Wrote {fname}")

print("\nDone. Generated 51 structures (000-050).")
