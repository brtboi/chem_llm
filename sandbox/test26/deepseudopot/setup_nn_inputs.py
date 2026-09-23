import shutil
import subprocess
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

START = 0
END = 10

NN_ROOT = Path("nn_inputs")

CALC_DIR = Path("calculations")

TEMPLATE_DIR = Path("nn_template")

QE_CONVERTER = "qe_bands_to_ref.py"

BANDS_SHIFTER = "shift.py"

OVERWRITE = False

ANG_TO_BOHR = 1.88973

# ============================================================
# HELPERS
# ============================================================

def parse_pw_input(filename):

    with open(filename) as f:
        lines = f.readlines()

    # --------------------------------------------------------
    # lattice
    # --------------------------------------------------------

    cell_idx = None

    for i, line in enumerate(lines):

        if "CELL_PARAMETERS" in line:

            cell_idx = i
            break

    if cell_idx is None:
        raise RuntimeError(f"No CELL_PARAMETERS in {filename}")

    lattice = []

    for i in range(3):

        vals = list(map(float, lines[cell_idx + 1 + i].split()))

        lattice.append(vals)

    # --------------------------------------------------------
    # scale + normalized cell
    # --------------------------------------------------------

    a_vec = lattice[0]

    scale = a_vec[0]

    normalized = []

    for vec in lattice:

        normalized.append([x / scale for x in vec])

    # --------------------------------------------------------
    # atomic positions
    # --------------------------------------------------------

    atom_idx = None

    for i, line in enumerate(lines):

        if "ATOMIC_POSITIONS crystal" in line:

            atom_idx = i
            break

    if atom_idx is None:
        raise RuntimeError(f"No ATOMIC_POSITIONS in {filename}")

    atoms = []

    counts = {}

    for line in lines[atom_idx + 1:]:

        stripped = line.strip()

        if stripped == "":
            break

        vals = stripped.split()

        species = vals[0]

        coords = vals[1:4]

        if species not in counts:
            counts[species] = 0

        label = f"{species}{counts[species]}"

        counts[species] += 1

        atoms.append((label, coords))

    return scale * ANG_TO_BOHR, normalized, atoms

# ============================================================
# WRITE system_0.par
# ============================================================

def write_system_par(outfile, scale, cell, atoms):

    with open(outfile, "w") as f:

        f.write(f"scale = {scale:.12f}\n\n")

        f.write("cell\n")

        for vec in cell:

            f.write(
                f"{vec[0]:.12f}\t"
                f"{vec[1]:.12f}\t"
                f"{vec[2]:.12f}\n"
            )

        f.write("\n")

        f.write("atoms\n")

        for label, coords in atoms:

            f.write(
                f"{label:<4} "
                f"{coords[0]} "
                f"{coords[1]} "
                f"{coords[2]}\n"
            )


# ============================================================
# MAIN LOOP
# ============================================================

for n in range(START, END + 1):

    prefix = f"{n:03d}"

    calc_path = CALC_DIR / prefix

    if not calc_path.exists():

        print(f"Missing calculation directory: {calc_path}")

        continue

    NN_ROOT.mkdir(exist_ok=True)

    for suffix in ["all", "g"]:
        nn_dir = NN_ROOT / f"inputs_{prefix}_{suffix}"

        # --------------------------------------------------------
        # overwrite handling
        # --------------------------------------------------------

        if nn_dir.exists():

            if OVERWRITE:

                print(f"Overwriting {nn_dir}")

                shutil.rmtree(nn_dir)

            else:

                print(f"Skipping existing {nn_dir}")

                continue

        # --------------------------------------------------------
        # copy template
        # --------------------------------------------------------

        print(f"Creating {nn_dir}")

        shutil.copytree(TEMPLATE_DIR, nn_dir)

        # --------------------------------------------------------
        # parse pw.in
        # --------------------------------------------------------

        pw_in = calc_path / "pw.in"

        scale, cell, atoms = parse_pw_input(pw_in)

        # --------------------------------------------------------
        # write system_0.par
        # --------------------------------------------------------

        write_system_par(
            nn_dir / "system_0.par",
            scale,
            cell,
            atoms
        )

        # --------------------------------------------------------
        # run QE converter
        # --------------------------------------------------------

        bands_file = f"{prefix}.bands.dat"

        print(f"Running QE converter for {prefix}")

        subprocess.run(
            [
                "python",
                f"../../{QE_CONVERTER}",
                bands_file
            ],
            cwd=calc_path
        )

        # --------------------------------------------------------
        # copy generated files
        # --------------------------------------------------------

        for fname in [
            "kpoints_0.par",
            "expBandStruct_0.par"
        ]:

            src = calc_path / fname

            if not src.exists():

                print(f"Missing generated file: {src}")

                continue

            shutil.copy(src, nn_dir / fname)

        # --------------------------------------------------------
        # run shift.py
        # --------------------------------------------------------

        print(f"Running shift.py in {nn_dir}")

        result = subprocess.run(
            [
                "python",
                f"{BANDS_SHIFTER}",
                "-5.25"
            ],
            cwd=nn_dir
        )

        if result.returncode != 0:

            print(f"shift.py failed in {nn_dir}")

        result = subprocess.run(
            ["mv",
            "expBandStruct_0_shift.par",
            "expBandStruct_0.par"],
            cwd=nn_dir
        )

        if result.returncode != 0:

            print(f"[mv expBandStruct_0_shift.par expBandStruct_0.par] failed in {nn_dir}")

        if suffix == "g":
            result = subprocess.run(
                [
                    "python",
                    "select_rows.py"
                ],
                cwd=nn_dir
            )

            if result.returncode != 0:

                print(f"select_rows.py failed in {nn_dir}")
    

print("Done.")
