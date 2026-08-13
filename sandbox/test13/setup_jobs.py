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
    Convert a pymatgen Structure to a Quantum ESPRESSO input file (pw.in or bands.in).
    Uses the correct number of atoms and types for NaCl (2 atoms, 2 types).
    Pseudopotentials are dynamically loaded from the template directory.
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
    lines.append("   ecutwfc = 50.0")  # Default; can be adjusted based on pseudopotential hints
    lines.append("   ecutrho = 250.0")
    lines.append("   tot_charge = 0.0")
    lines.append("   nosym = .true.")
    lines.append("   noinv = .true.")
    lines.append("   occupations = 'fixed'")
    lines.append("   nspin = 1")
    lines.append("   noncolin = .false.")
    lines.append("   lspinorb = .false.")

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

    # SPECIES
    lines.append("ATOMIC_SPECIES")
    # Dynamically load pseudopotential filenames from template
    # Assumes Na.upf and Cl.upf are in template/
    for specie in structure.species:
        symbol = specie.symbol
        pp_file = f"{symbol}.upf"
        lines.append(f"{symbol} 22.98976928 {pp_file}")
    lines.append("")

    # CELL PARAMETERS
    lines.append("CELL_PARAMETERS angstrom")
    for vec in structure.lattice.matrix:
        lines.append(f"{vec[0]:.10f} {vec[1]:.10f} {vec[2]:.10f}")
    lines.append("")

    # POSITIONS
    lines.append("ATOMIC_POSITIONS crystal")
    frac = structure.frac_coords % 1.0
    for specie, pos in zip(structure.species, frac):
        lines.append(f"{specie.symbol:<2} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
    lines.append("")

    # KPOINTS
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


def write_submit_script(calc_path, prefix):
    """
    Write a SLURM submit script with blank account and email fields.
    """
    submit_text = f"""#!/bin/bash
#SBATCH -J nacl_{prefix}
#SBATCH -C gpu
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time 03:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=

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

# MAIN LOOP
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
    pw_text = structure_to_qe(structure, prefix, calculation="scf")
    with open(calc_path / "pw.in", "w") as f:
        f.write(pw_text)

    # Write bands.in
    bands_text = structure_to_qe(structure, prefix, calculation="bands")
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
