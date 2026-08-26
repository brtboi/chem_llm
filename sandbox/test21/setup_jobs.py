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

# Pseudopotential filenames (from Pseudo-Dojo, nc, sr, pbesol, stringent, upf)
PP_TI = "Ti.pbesol-spn-rrkjus_psl.1.0.0.UPF"
PP_O = "O.pbesol-spn-rrkjus_psl.1.0.0.UPF"

# Energy cutoffs (Ry) from Pseudo-Dojo stringent recommendations
ECUTWFC = 42.0
ECUTRHO = 4 * ECUTWFC  # 168.0 Ry, default ratio for norm-conserving PPs

# Number of bands for band structure calculation
# TiO2 conventional cell: 4 Ti (4 val e- each) + 8 O (2 val e- each) = 32 val e- = 16 occupied bands
# nbnd=100 gives plenty of conduction bands
NBND = 100

# K-points for SCF (tetragonal rutile: a=b, c shorter)
KPOINTS_SCF = "8 8 4 0 0 0"

# High-symmetry k-path for rutile TiO2 (tetragonal):
# Gamma(0,0,0) -> X(0.5,0,0) -> M(0.5,0.5,0) -> Gamma(0,0,0) -> R(0.5,0.5,0.5) -> Z(0,0,0.5) -> X(0.5,0,0)
BANDS_PATH = [
    (0.0, 0.0, 0.0, 20),   # Gamma
    (0.5, 0.0, 0.0, 20),   # X
    (0.5, 0.5, 0.0, 20),   # M
    (0.0, 0.0, 0.0, 20),   # Gamma
    (0.5, 0.5, 0.5, 20),   # R
    (0.0, 0.0, 0.5, 20),   # Z
    (0.5, 0.0, 0.0, 1),    # X (endpoint, 1 point)
]


def structure_to_qe(structure, prefix, calculation):
    """
    Generate a Quantum ESPRESSO pw.x input file for the given structure.

    Parameters
    ----------
    structure : pymatgen Structure
        The crystal structure to write.
    prefix : str
        Unique prefix for output files (e.g. '000').
    calculation : str
        'scf' for ground-state calculation, 'bands' for band structure.
    """
    lines = []

    # &CONTROL
    lines.append("&CONTROL")
    lines.append(f"   prefix = '{prefix}'")
    lines.append(f"   calculation = '{calculation}'")
    lines.append("   restart_mode = 'from_scratch'")
    lines.append("   outdir = './'")
    lines.append("   wfcdir = './'")
    lines.append("   pseudo_dir = './'")
    lines.append("   verbosity = 'high'")
    lines.append("/")

    # &SYSTEM
    lines.append("&SYSTEM")
    # ibrav=0: no Bravais symmetry, use explicit CELL_PARAMETERS
    # (perturbed structures break tetragonal symmetry)
    lines.append("   ibrav = 0")
    # nat/ntyp computed from the actual structure object
    lines.append(f"   nat = {len(structure)}")
    lines.append(f"   ntyp = {len(structure.symbol_set)}")
    lines.append(f"   ecutwfc = {ECUTWFC}")
    lines.append(f"   ecutrho = {ECUTRHO}")
    lines.append("   tot_charge = 0.0")
    # TiO2 is a wide bandgap insulator (~3 eV), use fixed occupations
    lines.append("   occupations = 'fixed'")
    # No spin polarization: Ti4+ is d0, non-magnetic
    # No SOC: not needed for this study
    if calculation == "bands":
        lines.append(f"   nbnd = {NBND}")
    lines.append("/")

    # &ELECTRONS
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
    lines.append(f"Ti 47.867 {PP_TI}")
    lines.append(f"O 15.999 {PP_O}")
    lines.append("")

    # CELL_PARAMETERS (in Angstrom)
    lines.append("CELL_PARAMETERS angstrom")
    for vec in structure.lattice.matrix:
        lines.append(f"{vec[0]:.10f} {vec[1]:.10f} {vec[2]:.10f}")
    lines.append("")

    # ATOMIC_POSITIONS (fractional coordinates)
    lines.append("ATOMIC_POSITIONS crystal")
    frac = structure.frac_coords % 1.0
    for specie, pos in zip(structure.species, frac):
        lines.append(f"{specie.symbol:<2} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
    lines.append("")

    # K_POINTS
    if calculation == "scf":
        lines.append("K_POINTS automatic")
        lines.append(KPOINTS_SCF)
    elif calculation == "bands":
        lines.append("K_POINTS crystal_b")
        lines.append(str(len(BANDS_PATH)))
        for (kx, ky, kz, npts) in BANDS_PATH:
            lines.append(f"{kx:.6f} {ky:.6f} {kz:.6f} {npts}")

    return "\n".join(lines)


def write_submit_script(calc_path, prefix):
    """Write the SLURM submission script for a calculation directory."""
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

    # Copy template (pseudopotentials) to calc directory
    if calc_path.exists():
        if OVERWRITE:
            print(f"Overwriting {calc_path}")
            shutil.rmtree(calc_path)
        else:
            print(f"Skipping existing directory: {calc_path}")
            continue

    shutil.copytree(TEMPLATE_DIR, calc_path)

    # Load structure from CIF
    structure = Structure.from_file(cif_file)

    # Write pw.in (SCF)
    pw_text = structure_to_qe(structure, prefix, calculation="scf")
    with open(calc_path / "pw.in", "w") as f:
        f.write(pw_text)

    # Write bands.in (band structure)
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
