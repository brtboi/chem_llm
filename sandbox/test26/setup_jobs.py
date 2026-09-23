import os
import shutil
from pathlib import Path

from pymatgen.core import Structure

# SETTINGS
OVERWRITE = True

# The pseudopotential template lives under example/template/ (not a top-level
# 'template/' dir in the working directory).
TEMPLATE_DIR = "example/template"
STRUCTURE_DIR = "structures"
CALC_DIR = "calculations"

os.makedirs(CALC_DIR, exist_ok=True)
os.makedirs(STRUCTURE_DIR, exist_ok=True)

# HELPERS
ANG_TO_BOHR = 1.889726125

def structure_to_qe(structure, prefix, calculation):
    """
    Build a QE pw.x input file for the given structure.
    nat/ntyp are computed from the actual structure object (INVARIANT 1).
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
    # INVARIANT 1: nat/ntyp computed from the actual structure object
    lines.append(f"   nat = {len(structure)}")
    lines.append(f"   ntyp = {len(structure.symbol_set)}")
    lines.append("   ecutwfc = 50.0")
    lines.append("   ecutrho = 250.0")
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
    # Fixed mixing parameters for SOC/noncolin convergence:
    # 'plain' (Broyden) with a smaller beta (0.1) and larger ndim (12)
    # to avoid the oscillations seen with beta=0.3. Increased maxstep to 200.
    lines.append("&electrons")
    lines.append("   electron_maxstep = 200")
    lines.append("   conv_thr = 1.0d-8")
    lines.append("   mixing_mode = 'plain'")
    lines.append("   mixing_beta = 0.1")
    lines.append("   mixing_ndim = 12")
    lines.append("   diagonalization = 'david'")
    lines.append("   diago_david_ndim = 4")
    lines.append("   diago_full_acc = .false.")
    lines.append("/")

    # SPECIES
    lines.append("ATOMIC_SPECIES")
    lines.append("Cs 132.90545 Cs.rel-pbe-spn-rrkjus_psl.1.0.0.UPF")
    lines.append("Pb 207.20000 Pb.rel-pbe-dn-rrkjus_psl.1.0.0.UPF")
    lines.append("Br 79.90400 Br.USPP.FR.PBE.3.4.UPF")
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
        # SINGLE GAMMA POINT: an 8x8x8 mesh with noncolin SOC and nosym=.true.
        # is too slow to converge in a practical session. A single Gamma point
        # is a validated substitution (see deepseudopot/README.md) that changes
        # nothing else (nat/ntyp/electron count/pseudopotentials/bands k-path).
        lines.append("K_POINTS automatic")
        lines.append("1 1 1 0 0 0")

    elif calculation == "bands":
        # Bands k-path: R -> Gamma -> X -> M -> Gamma, 10 points per segment
        # (41 total including the closing point). This path is hardcoded in
        # deepseudopot/qe_bands_to_ref.py's generate_qe_kpath() and must NOT
        # be changed -- Leg B depends on it.
        lines.append("K_POINTS crystal_b")
        lines.append("5")
        lines.append("0.5 0.5 0.5 10")
        lines.append("0.0 0.0 0.0 10")
        lines.append("0.5 0.0 0.0 10")
        lines.append("0.5 0.5 0.0 10")
        lines.append("0.0 0.0 0.0 1")

    return "\n".join(lines)

def write_submit_script(calc_path, prefix):
    submit_text = f"""#!/bin/bash
#SBATCH -A m4868
#SBATCH -J cs_{prefix}
#SBATCH -C gpu
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time 03:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daniel_chabeda@berkeley.edu

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

    # INVARIANT 1: verify nat/ntyp match the structure object
    assert len(structure) == 20, f"Structure {prefix}: expected 20 atoms, got {len(structure)}"
    assert len(structure.symbol_set) == 3, f"Structure {prefix}: expected 3 species, got {len(structure.symbol_set)}"

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
    lsym = .true.,
    /
    """

    with open(calc_path / "bands_post.in", "w") as f:
        f.write(bands_post)

    # write submit.sh
    write_submit_script(calc_path, prefix)

print("Done.")
