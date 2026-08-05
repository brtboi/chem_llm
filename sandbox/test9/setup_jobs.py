import os
import shutil
from pathlib import Path

from pymatgen.core import Structure

# SETTINGS
OVERWRITE = False

TEMPLATE_DIR = "template"
STRUCTURE_DIR = "structures"
CALC_DIR = "calculations"

os.makedirs(CALC_DIR, exist_ok=True)
os.makedirs(STRUCTURE_DIR, exist_ok=True)

# HELPERS
ANG_TO_BOHR = 1.889726125

def structure_to_qe(structure, prefix, calculation):
    """
    Convert a pymatgen Structure to a Quantum ESPRESSO input file.
    
    Parameters:
    - structure: pymatgen Structure object
    - prefix: string prefix for the calculation
    - calculation: string, either 'scf' or 'bands'
    
    Returns:
    - string containing the full input file content
    """
    a, b, c = structure.lattice.abc

    a_bohr = a * ANG_TO_BOHR
    b_bohr = b * ANG_TO_BOHR
    c_bohr = c * ANG_TO_BOHR

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
    lines.append("   ecutwfc = 60.0")  # Increased for TiO2
    lines.append("   ecutrho = 480.0")  # Increased for TiO2
    lines.append("   tot_charge = 0.0")
    lines.append("   nosym = .true.")
    lines.append("   noinv = .true.")
    lines.append("   occupations = 'fixed'")
    lines.append("   nspin = 1")  # Non-magnetic calculation
    lines.append("   noncolin = .false.")
    lines.append("   lspinorb = .false.")
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

    # SPECIES
    lines.append("ATOMIC_SPECIES")
    # Use standard PBE pseudopotentials for Ti and O
    lines.append("Ti  47.867  Ti.pbe-n-rrkjus_psl.1.0.0.UPF")
    lines.append("O   15.999  O.pbe-n-rrkjus_psl.1.0.0.UPF")
    lines.append("")

    # CELL PARAMETERS
    lines.append("CELL_PARAMETERS angstrom")

    for vec in structure.lattice.matrix:
        lines.append(
            f"{vec[0]:.10f} "
            f"{vec[1]:.10f} "
            f"{vec[2]:.10f}"
        )

    lines.append("")

    # POSITIONS
    lines.append("ATOMIC_POSITIONS crystal")

    frac = structure.frac_coords % 1.0

    for specie, pos in zip(structure.species, frac):
        lines.append(
            f"{specie.symbol:<2} "
            f"{pos[0]:.6f} "
            f"{pos[1]:.6f} "
            f"{pos[2]:.6f}"
        )

    lines.append("")

    # KPOINTS
    if calculation == "scf":
        lines.append("K_POINTS automatic")
        lines.append("6 6 6 0 0 0")  # Reduced grid for efficiency
    elif calculation == "bands":
        lines.append("K_POINTS crystal_b")
        lines.append("5")
        lines.append("0.5 0.5 0.5 10")
        lines.append("0.0 0.0 0.0 10")
        lines.append("0.5 0.0 0.0 10")
        lines.append("0.5 0.5 0.0 10")
        lines.append("0.0 0.0 0.0 1")

    return "\n".join(lines)

def write_submit_script(calc_path, prefix):
    """
    Write a SLURM submit script with blank account and email fields.
    """
    submit_text = f"""#!/bin/bash
#SBATCH -A 
#SBATCH -J ti_{prefix}
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time 01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

module load espresso

PW=pw.x

echo "Running SCF for {prefix}"

srun -n 1 $PW -in pw.in > pw.out

if [ $? -ne 0 ]; then
    echo "SCF failed"
    exit 1
fi

echo "Finished {prefix}"
"""

    with open(calc_path / "submit.sh", "w") as f:
        f.write(submit_text)

# MAIN LOOP
cif_files = sorted(Path(STRUCTURE_DIR).glob("*.cif"))

for n, cif_file in enumerate(cif_files):
    prefix = f"{n:03d}"
    calc_path = Path(CALC_DIR) / prefix

    print("Setting up", calc_path)

    # copy template
    if calc_path.exists():
        if OVERWRITE:
            print(f"Overwriting {calc_path}")
            shutil.rmtree(calc_path)
        else:
            print(f"Skipping existing directory: {calc_path}")
            continue
    
    shutil.copytree(TEMPLATE_DIR, calc_path)

    # load structure
    structure = Structure.from_file(cif_file)

    # write pw.in (only scf, no bands)
    pw_text = structure_to_qe(
        structure,
        prefix,
        calculation="scf"
    )

    with open(calc_path / "pw.in", "w") as f:
        f.write(pw_text)

    # write submit.sh
    write_submit_script(calc_path, prefix)

print("Done.")
