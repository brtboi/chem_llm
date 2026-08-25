import os
import shutil
from pathlib import Path
from pymatgen.core import Structure

# SETTINGS
OVERWRITE = False

TEMPLATE_DIR = "template"
STRUCTURE_DIR = "structures"
CALC_DIR = "calculations"

# Ensure directories exist
os.makedirs(CALC_DIR, exist_ok=True)
os.makedirs(STRUCTURE_DIR, exist_ok=True)

# Constants
ANG_TO_BOHR = 1.889726125

# Pseudopotential recommendations from Pseudo-Dojo (in Ha)
# Converted to Ry: 1 Ha = 0.5485734905362112 Ry
# Cs: 25.0 Ha = 13.7143 Ry
# Pb: 28.0 Ha = 15.3600 Ry
# Br: 23.0 Ha = 12.6172 Ry
# Use highest: 15.3600 Ry for ecutwfc
# ecutrho = 4 * ecutwfc = 61.4400 Ry
ECUTWFC = 15.3600  # Ry
ECUTRHO = 4 * ECUTWFC  # Ry

# Atomic masses (g/mol)
ATOMIC_MASSES = {
    "Cs": 132.90545,
    "Pb": 207.2,
    "Br": 79.904
}

# Pseudopotential filenames (must match downloaded files)
PSEUDO_FILES = {
    "Cs": "Cs.upf",
    "Pb": "Pb.upf",
    "Br": "Br.upf"
}

# Helper function to convert structure to Quantum ESPRESSO input
def structure_to_qe(structure, prefix, calculation):
    # Extract lattice parameters in Bohr
    a, b, c = structure.lattice.abc
    a_bohr = a * ANG_TO_BOHR
    b_bohr = b * ANG_TO_BOHR
    c_bohr = c * ANG_TO_BOHR

    # celldm1 = a (in Bohr), celldm2 = b/a, celldm3 = c/a
    celldm1 = a_bohr
    celldm2 = b_bohr / a_bohr
    celldm3 = c_bohr / a_bohr

    lines = []

    # CONTROL
    lines.append("&CONTROL")
    lines.append(f"   prefix = '{prefix}'")
    lines.append(f"   calculation = '{calculation}'")
    lines.append("   restart_mode = 'from_scratch'")
    lines.append("   outdir = './'")
    lines.append("   wfcdir = './'")
    lines.append("   pseudo_dir = './'")
    lines.append("   verbosity = 'high'")
    lines.append("/")

    # SYSTEM
    lines.append("&SYSTEM")
    lines.append("   ibrav = 0")
    lines.append(f"   nat = {len(structure)}")
    lines.append(f"   ntyp = {len(set(site.specie.symbol for site in structure))}")
    lines.append(f"   ecutwfc = {ECUTWFC}")
    lines.append(f"   ecutrho = {ECUTRHO}")
    lines.append("   tot_charge = 0.0")
    lines.append("   nosym = .true.")
    lines.append("   noinv = .true.")
    lines.append("   occupations = 'fixed'")
    lines.append("   nspin = 4")
    lines.append("   noncolin = .true.")
    lines.append("   lspinorb = .true.")

    if calculation == "bands":
        lines.append("   nbnd = 200")

    lines.append("/")

    # ELECTRONS
    lines.append("&electrons")
    lines.append("   electron_maxstep = 100")
    lines.append("   conv_thr = 1.0d-8")
    lines.append("   mixing_mode = 'plain'")
    lines.append("   mixing_beta = 0.3")
    lines.append("   mixing_ndim = 8")
    lines.append("   diagonalization = 'david'")
    lines.append("   diago_david_ndim = 4")
    lines.append("   diago_full_acc = .false.")
    lines.append("/")

    # ATOMIC_SPECIES
    lines.append("ATOMIC_SPECIES")
    for element in ["Cs", "Pb", "Br"]:
        mass = ATOMIC_MASSES[element]
        pseudo_file = PSEUDO_FILES[element]
        lines.append(f"{element} {mass} {pseudo_file}")
    lines.append("")

    # CELL_PARAMETERS
    lines.append("CELL_PARAMETERS angstrom")
    for vec in structure.lattice.matrix:
        lines.append(f"{vec[0]:.10f} {vec[1]:.10f} {vec[2]:.10f}")
    lines.append("")

    # ATOMIC_POSITIONS
    lines.append("ATOMIC_POSITIONS crystal")
    frac = structure.frac_coords % 1.0
    for specie, pos in zip(structure.species, frac):
        lines.append(f"{specie.symbol:<2} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
    lines.append("")

    # K_POINTS
    if calculation == "scf":
        lines.append("K_POINTS automatic")
        lines.append("8 8 8 0 0 0")
    elif calculation == "bands":
        lines.append("K_POINTS crystal_b")
        lines.append("5")
        lines.append("0.5 0.5 0.5 10")
        lines.append("0.0 0.0 0.0 10")
        lines.append("0.5 0.0 0.0 10")
        lines.append("0.5 0.5 0.0 10")
        lines.append("0.0 0.0 0.0 1")

    return "\n".join(lines)

# Helper function to write submit.sh script
def write_submit_script(calc_path, prefix):
    submit_text = f"""#!/bin/bash
#SBATCH -A m4735
#SBATCH -J cs_{prefix}
#SBATCH -C gpu
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time 03:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=brent.hu@yale.edu

module load espresso

PW=pw.x
BANDS=bands.x

echo "Running SCF for {prefix}"

srun -n 8 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 $PW -in pw.in > pw.out

if [ $? -ne 0 ]; then
    echo "SCF failed"
    exit 1
fi

echo "Running bands SCF for {prefix}"

srun -n 8 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 $PW -in bands.in > bands_pw.out

if [ $? -ne 0 ]; then
    echo "Bands calculation failed"
    exit 1
fi

echo "Running bands.x for {prefix}"

$BANDS -in bands_post.in > bands_post.out

echo "Finished {prefix}"
"""

    with open(calc_path / "submit.sh", "w") as f:
        f.write(submit_text)

# Main loop
if __name__ == "__main__":
    cif_files = sorted(Path(STRUCTURE_DIR).glob("*.cif"))

    for n, cif_file in enumerate(cif_files):
        prefix = f"{n:03d}"
        calc_path = Path(CALC_DIR) / prefix

        print(f"Setting up {calc_path}")

        # Copy template
        if calc_path.exists():
            if OVERWRITE:
                print(f"Overwriting {calc_path}")
                shutil.rmtree(calc_path)
            else:
                print(f"Skipping existing directory: {calc_path}")
                continue
        shutil.copytree(TEMPLATE_DIR, calc_path)

        # Load structure
        structure = Structure.from_file(cif_file)

        # Write pw.in
        pw_text = structure_to_qe(
            structure,
            prefix,
            calculation="scf"
        )
        with open(calc_path / "pw.in", "w") as f:
            f.write(pw_text)

        # Write bands.in
        bands_text = structure_to_qe(
            structure,
            prefix,
            calculation="bands"
        )
        with open(calc_path / "bands.in", "w") as f:
            f.write(bands_text)

        # Write bands_post.in
        bands_post = f"""&BANDS
    prefix  = '{prefix}'
    outdir  = './'
    filband = '{prefix}.bands.dat'
    lsym = .true.,
    /
"""
        with open(calc_path / "bands_post.in", "w") as f:
            f.write(bands_post)

        # Write submit.sh
        write_submit_script(calc_path, prefix)

    print("Done.")
