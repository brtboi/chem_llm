import os
import shutil
from pathlib import Path

from pymatgen.core import Structure

# SETTINGS
OVERWRITE = False

# Use paths relative to this script's directory so it works regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(SCRIPT_DIR, "template")
STRUCTURE_DIR = os.path.join(SCRIPT_DIR, "structures")
CALC_DIR = os.path.join(SCRIPT_DIR, "calculations")

os.makedirs(CALC_DIR, exist_ok=True)
os.makedirs(STRUCTURE_DIR, exist_ok=True)

# Pseudopotential filenames (from Pseudo-Dojo, nc, sr, pbesol, stringent, upf)
# Ti: ecut = 46.0 Ry, O: ecut = 48.0 Ry -> use max = 48.0 Ry
PSEUDO_FILES = {
    "Ti": "Ti.pbesol-spn-rrkjus_psl.1.0.0.UPF",
    "O":  "O.pbesol-spn-rrkjus_psl.1.0.0.UPF",
}

# Atomic masses (amu)
ATOMIC_MASSES = {
    "Ti": 47.867,
    "O":  15.999,
}

# Energy cutoffs (Ry)
ECUTWFC = 48.0   # max of Ti (46.0) and O (48.0) from Pseudo-Dojo stringent hints
ECUTRHO = 192.0  # 4x ecutwfc, standard for charge density

# Number of bands to compute (32 occupied + 68 unoccupied for the 12-atom cell)
NBND = 100


def structure_to_qe(structure, prefix, calculation):
    """Generate a Quantum ESPRESSO pw.x input file for the given structure.
    
    Args:
        structure: pymatgen Structure object
        prefix: output file prefix (e.g. '000')
        calculation: 'scf' or 'bands'
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
    lines.append("&SYSTEM")
    lines.append("   ibrav = 0")
    # nat/ntyp MUST be computed from the structure object, never hardcoded
    lines.append(f"   nat = {len(structure)}")
    lines.append(f"   ntyp = {len(structure.symbol_set)}")
    lines.append(f"   ecutwfc = {ECUTWFC:.1f}")
    lines.append(f"   ecutrho = {ECUTRHO:.1f}")
    lines.append("   tot_charge = 0.0")
    lines.append("   nosym = .true.")
    lines.append("   noinv = .true.")
    lines.append("   occupations = 'fixed'")
    lines.append("   nspin = 1")

    if calculation == "bands":
        lines.append(f"   nbnd = {NBND}")

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
    for symbol in sorted(structure.symbol_set):
        mass = ATOMIC_MASSES[symbol]
        pp_file = PSEUDO_FILES[symbol]
        lines.append(f"{symbol} {mass:.3f} {pp_file}")
    lines.append("")

    # CELL_PARAMETERS (in Angstrom)
    lines.append("CELL_PARAMETERS angstrom")
    for vec in structure.lattice.matrix:
        lines.append(f"{vec[0]:.10f} {vec[1]:.10f} {vec[2]:.10f}")
    lines.append("")

    # ATOMIC_POSITIONS (fractional/crystal coordinates)
    lines.append("ATOMIC_POSITIONS crystal")
    frac = structure.frac_coords % 1.0
    for specie, pos in zip(structure.species, frac):
        lines.append(f"{specie.symbol:<2} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
    lines.append("")

    # K_POINTS
    if calculation == "scf":
        # Gamma-centered Monkhorst-Pack grid for SCF
        lines.append("K_POINTS automatic")
        lines.append("8 8 8 0 0 0")
    elif calculation == "bands":
        # High-symmetry path for tetragonal rutile TiO2:
        # Gamma(0,0,0) -> X(0.5,0,0) -> M(0.5,0.5,0) -> R(0.5,0.5,0.5) -> Gamma(0,0,0)
        lines.append("K_POINTS crystal_b")
        lines.append("5")
        lines.append("0.0 0.0 0.0 10")
        lines.append("0.5 0.0 0.0 10")
        lines.append("0.5 0.5 0.0 10")
        lines.append("0.5 0.5 0.5 10")
        lines.append("0.0 0.0 0.0 1")

    return "\n".join(lines)


def write_submit_script(calc_path, prefix):
    """Write a SLURM submit script for the given calculation directory."""
    submit_text = f"""#!/bin/bash
#SBATCH -A m4735
#SBATCH -J tio2_{prefix}
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


# MAIN LOOP
cif_files = sorted(Path(STRUCTURE_DIR).glob("structure_*.cif"))

for n, cif_file in enumerate(cif_files):
    prefix = f"{n:03d}"
    calc_path = Path(CALC_DIR) / prefix

    print(f"Setting up {calc_path}")

    # Copy template (pseudopotentials) into calc directory
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

    # Write pw.in (SCF)
    pw_text = structure_to_qe(structure, prefix, calculation="scf")
    with open(calc_path / "pw.in", "w") as f:
        f.write(pw_text)

    # Write bands.in (bands SCF)
    bands_text = structure_to_qe(structure, prefix, calculation="bands")
    with open(calc_path / "bands.in", "w") as f:
        f.write(bands_text)

    # Write bands_post.in (bands.x post-processing)
    bands_post = f"""&BANDS
   prefix  = '{prefix}'
   outdir  = './'
   filband = '{prefix}.bands.dat'
   lsym = .true.
/
"""
    with open(calc_path / "bands_post.in", "w") as f:
        f.write(bands_post)

    # Write submit.sh
    write_submit_script(calc_path, prefix)

print("Done.")
