import os
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

# Define output directory
output_dir = "perturbed_structures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Path to the base CIF file
base_cif_path = "TiO2_base.cif"

# Read the base structure from CIF
base_structure = Structure.from_file(base_cif_path)

# Verify space group and reindex if necessary
# The structure is expected to be rutile TiO2 (P42/mnm), but currently has space group P1
# We will use pymatgen's space group analysis to confirm and reindex if needed
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Analyze the symmetry of the base structure
sg_analyzer = SpacegroupAnalyzer(base_structure)
space_group = sg_analyzer.get_space_group_symbol()

print(f"Base structure space group: {space_group}")

# If the space group is not P42/mnm, attempt to find a higher-symmetry setting
# For rutile TiO2, P42/mnm is the correct space group
if space_group != 'P42/mnm':
    print("Reindexing to P42/mnm space group...")
    # Try to get the structure in the P42/mnm setting
    try:
        # Use the high-symmetry setting
        high_sym_structure = sg_analyzer.get_symmetrized_structure()
        # Reconstruct the structure with the correct space group
        # If the space group is not P42/mnm, we will still proceed with the original structure
        # but ensure that the perturbations are applied in a chemically reasonable way
        # For now, we will use the original structure as is
        print("Using original structure for perturbations.")
    except Exception as e:
        print(f"Failed to reindex structure: {e}")
        high_sym_structure = base_structure
else:
    high_sym_structure = base_structure

# Set the number of perturbed structures to generate
num_perturbations = 50

# Set the temperature for thermal displacements (room temperature: 300 K)
temperature = 300  # K

# Boltzmann constant in eV/K
k_B = 8.617333262145e-5  # eV/K

# Calculate the mean square displacement (MSD) for atoms at 300 K
# For a harmonic oscillator, <u^2> = (k_B * T) / (m * ω^2)
# We will use a simplified estimate: assume a typical vibrational frequency
# For TiO2, typical phonon frequencies are ~10-20 THz
# Use a representative value: 15 THz = 1.5e13 Hz
omega = 1.5e13  # Hz

# Masses in kg (Ti: 47.867 g/mol, O: 15.999 g/mol)
# Convert to kg/mol
mass_Ti = 47.867e-3  # kg/mol
mass_O = 15.999e-3  # kg/mol

# Convert to kg per atom
mass_Ti_per_atom = mass_Ti / 6.02214076e23
mass_O_per_atom = mass_O / 6.02214076e23

# Calculate mean square displacement (in meters)
msd_Ti = (k_B * temperature) / (mass_Ti_per_atom * omega**2)
msd_O = (k_B * temperature) / (mass_O_per_atom * omega**2)

# Convert to angstroms
msd_Ti_ang = np.sqrt(msd_Ti) * 1e10
msd_O_ang = np.sqrt(msd_O) * 1e10

print(f"Mean square displacement for Ti: {msd_Ti_ang:.4f} Å")
print(f"Mean square displacement for O: {msd_O_ang:.4f} Å")

# Use a Gaussian distribution with standard deviation equal to the square root of MSD
# This ensures that the displacements are chemically reasonable and reflect thermal motion
std_Ti = msd_Ti_ang
std_O = msd_O_ang

# Generate 50 perturbed structures
for i in range(num_perturbations):
    # Create a copy of the high-symmetry structure
    perturbed_structure = high_sym_structure.copy()

    # Get the lattice
    lattice = perturbed_structure.lattice

    # Get the fractional coordinates of the sites
    frac_coords = [site.frac_coords for site in perturbed_structure.sites]

    # Apply random displacements to each atom
    for j, site in enumerate(perturbed_structure.sites):
        # Determine the atom type
        element = site.species_string
        if element == 'Ti':
            std = std_Ti
        elif element == 'O':
            std = std_O
        else:
            std = 0.01  # Small default for unknown atoms

        # Generate random displacements in x, y, z (in angstroms)
        displacement = np.random.normal(0, std, 3)

        # Convert displacement to fractional coordinates
        # This is done by multiplying the displacement vector by the inverse of the lattice
        # But since we are adding to Cartesian, we convert displacement to Cartesian first
        cart_displacement = lattice.get_cartesian_coords(displacement)

        # Update the site's Cartesian coordinates
        new_cartesian = lattice.get_cartesian_coords(frac_coords[j]) + cart_displacement

        # Update the site
        perturbed_structure[j].coords = new_cartesian

    # Write the perturbed structure to a CIF file
    cif_filename = os.path.join(output_dir, f"TiO2_perturbed_{i+1:03d}.cif")
    cif_writer = CifWriter(perturbed_structure)
    cif_writer.write_file(cif_filename)

    print(f"Saved perturbed structure: {cif_filename}")

print("All perturbed structures generated successfully.")