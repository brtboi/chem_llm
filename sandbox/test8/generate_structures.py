import os
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter

# SETTINGS
N_STRUCTURES = 50

# Directory for output structures
os.makedirs("structures", exist_ok=True)

# Temperature in Kelvin for thermal displacement (room temperature ~300 K)
TEMP_K = 300

# Boltzmann constant in eV/K
BOLTZMANN = 8.617333262145e-5

# Atomic masses (in amu)
MASS_Ti = 47.867
MASS_O = 15.999

# Convert temperature to thermal energy (in eV)
thermal_energy = BOLTZMANN * TEMP_K

# Estimate root-mean-square displacement (in angstroms) using Einstein model:
# <u^2> = (kT / (m * omega^2))
# Approximate omega from Debye model or typical phonon frequencies
# For TiO2 rutile, typical phonon frequencies are ~10-20 THz
# Use a rough estimate: omega ~ 15 THz = 15e12 Hz
# omega = 2*pi*f
# kT = 0.02585 eV at 300K
# m = mass in kg
# Convert to amu to kg: 1 amu = 1.66053906660e-27 kg
# Use approximate Debye temperature for TiO2: ~500 K
# <u^2> = (kT / (m * omega^2))
# But we use a simpler empirical estimate: ~0.05 - 0.15 Å RMS displacement
# Use 0.1 Å RMS as a reasonable estimate for room temperature
RMS_DISPLACEMENT = 0.1  # in angstroms

# Read the base structure from CIF
base_structure = Structure.from_file("rutile_tio2.cif")

# Function to apply chemically reasonable random displacements
# based on thermal motion at room temperature
def perturb_structure(structure, rms_displacement=RMS_DISPLACEMENT):
    """
    Apply random displacements to atoms based on thermal motion at room temperature.
    Uses a Gaussian distribution with standard deviation equal to the RMS displacement.
    Ensures displacements are physically reasonable and maintain chemical integrity.
    """
    s = structure.copy()

    # For each atom, generate a random displacement vector
    # with magnitude drawn from a Gaussian distribution
    # centered at 0, with standard deviation = rms_displacement
    for i, site in enumerate(s):
        # Use a Gaussian random vector with standard deviation = rms_displacement
        # in each Cartesian direction
        dr = np.random.normal(scale=rms_displacement, size=3)

        # Apply displacement in Cartesian coordinates
        s.translate_sites(i, dr, frac_coords=False)

    return s

# Generate 50 perturbed structures
for n in range(N_STRUCTURES):
    # Create a perturbed copy of the base structure
    perturbed_structure = perturb_structure(base_structure)

    # Write to CIF file
    filename = f"structures/structure_{n:03d}.cif"
    CifWriter(perturbed_structure).write_file(filename, mode='wt')
    print(f"Wrote {filename}")

# Optional: Save the base structure as a reference
base_filename = "structures/base_structure.cif"
CifWriter(base_structure).write_file(base_filename, mode='wt')
print(f"Wrote {base_filename}")
