import os
import shutil
from pathlib import Path

from pymatgen.core import Structure

# SETTINGS
OVERWRITE = False

# Use script-relative paths so the script works regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(SCRIPT_DIR, "template")
STRUCTURE_DIR = os.path.join(SCRIPT_DIR, "structures")
CALC_DIR = os.path.join(SCRIPT_DIR, "calculations")

os.makedirs(CALC_DIR, exist_ok=True)
os.makedirs(STRUCTURE_DIR, exist_ok=True)

# HELPERS
ANG_TO_BOHR = 1.889726125

# Pseudopotential files (norm-conserving, scalar-relativistic, PBEsol, stringent, UPF)
TI_PSP = "Ti.pbesol-spn-rrkjus_psl.1.0.0.UPF"
O_PSP = "O.pbesol-spn-rrkjus_psl.1.0.0.UPF"

# Energy cutoffs (in Ry) from Pseudo-Dojo recommendations (high accuracy level)
# Ti: 46.0 Ry, O: 48.0 Ry -> use max for ecutwfc
ECUTWFC = 48.0
ECUTRHO = 240.0  # 5x ecutwfc, standard for charge density


def structure_to_qe(structure, prefix, calculation):
    """
    Generate a Quantum ESPRESSO pw.x input file for the given structure.
    
    Parameters:
        structure: pymatgen Structure object
        prefix: unique identifier for output files
        calculation: 'scf' for ground state, 'bands' for band structure
    """
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
    # nat = 4 (2 Ti + 2 O in rutile unit cell)
    # ntyp = 2 (Ti and O)
    # ibrav = 0: no symmetry, use explicit cell parameters (needed for perturbed structures)
    # nosym = .true.: disable symmetry (perturbed structures break symmetry)
    # noinv = .true.: disable inversion symmetry
    # nspin = 1: spin-unpolarized (TiO2 is a non-magnetic insulator)
    # No lspinorb, no noncolin: Ti and O are light elements, spin-orbit coupling negligible
    # occupations = 'fixed': for insulator, fixed occupations
    lines.append("&SYSTEM")
    lines.append("   ibrav = 0")
    lines.append("   nat = 4")
    lines.append("   ntyp = 2")
    lines.append(f"   ecutwfc = {ECUTWFC}")
    lines.append(f"   ecutrho = {ECUTRHO}")
    lines.append("   tot_charge = 0.0")
    lines.append("   nosym = .true.")
    lines.append("   noinv = .true.")
    lines.append("   occupations = 'fixed'")
    lines.append("   nspin = 1")

    if calculation == "bands":
        # nbnd = 40: enough to cover valence + conduction bands (8 occupied + 32 unoccupied)
        lines.append("   nbnd = 40")

    lines.append("/")

    # ELECTRONS
    lines.append("&ELECTRONS")
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
    lines.append(f"Ti 47.867 {TI_PSP}")
    lines.append(f"O 15.999 {O_PSP}")
    lines.append("")

    # CELL_PARAMETERS angstrom
    lines.append("CELL_PARAMETERS angstrom")
    for vec in structure.lattice.matrix:
        lines.append(
            f"{vec[0]:.10f} "
            f"{vec[1]:.10f} "
            f"{vec[2]:.10f}"
        )
    lines.append("")

    # ATOMIC_POSITIONS crystal
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

    # K_POINTS
    if calculation == "scf":
        # Automatic k-mesh for SCF: 8x8x4 for tetragonal rutile TiO2
        # Denser in a/b directions than c due to tetragonal symmetry
        lines.append("K_POINTS automatic")
        lines.append("8 8 4 0 0 0")

    elif calculation == "bands":
        # Crystal B path through high-symmetry points for rutile TiO2 (tetragonal)
        # Gamma(0,0,0) -> X(0.5,0,0) -> M(0.5,0.5,0) -> Gamma(0,0,0) -> R(0.5,0.5,0.5)
        lines.append("K_POINTS crystal_b")
        lines.append("5")
        lines.append("0.0 0.0 0.0 10")
        lines.append("0.5 0.0 0.0 10")
        lines.append("0.5 0.5 0.0 10")
        lines.append("0.0 0.0 0.0 10")
        lines.append("0.5 0.5 0.5 1")

    return "\n".join(lines)


def write_submit_script(calc_path, prefix):
    """Write a SLURM submit script for the given calculation directory."""
    submit_text = f"""#!/bin/bash
#SBATCH -A m4735
#SBATCH -J tio2_{prefix}
#SBATCH -C gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time 03:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=brent.hu@yale.edu

module load espresso

PW=pw.x
BANDS=bands.x

echo "Running SCF for {prefix}"

srun -n 4 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 $PW -in pw.in > pw.out

if [ $? -ne 0 ]; then
    echo "SCF failed"
    exit 1
fi

echo "Running bands SCF for {prefix}"

srun -n 4 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 $PW -in bands.in > bands_pw.out

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
# Only pick up structure_*.cif files (not the base rutile_TiO2_base.cif)
cif_files = sorted(Path(STRUCTURE_DIR).glob("structure_*.cif"))

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

    # write pw.in
    pw_text = structure_to_qe(
        structure,
        prefix,
        calculation="scf"
    )

    with open(calc_path / "pw.in", "w") as f:
        f.write(pw_text)

    # write bands.in
    bands_text = structure_to_qe(
        structure,
        prefix,
        calculation="bands"
    )

    with open(calc_path / "bands.in", "w") as f:
        f.write(bands_text)

    # write bands_post.in
    bands_post = f"""&BANDS
    prefix  = '{prefix}'
    outdir  = './'
    filband = '{prefix}.bands.dat'
    lsym = .true.
/
"""

    with open(calc_path / "bands_post.in", "w") as f:
        f.write(bands_post)

    # write submit.sh
    write_submit_script(calc_path, prefix)

print("Done.")
