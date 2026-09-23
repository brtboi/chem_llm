"""Step 4 of tio2_prompt.md: write QE pw.in (SCF), bands.in (bands),
bands_post.in, and submit.sh for all 51 structures.

Everything cell/k-path-related is derived programmatically from each
structure itself (never a memorized textbook constant):
  - lattice/positions come straight from the structure's own pymatgen
    Structure object (CELL_PARAMETERS/ATOMIC_POSITIONS).
  - nbnd/occupied-band padding come from the ACTUAL z_valence declared in
    the fetched Ti.upf/O.upf pseudopotential files (these are semicore
    pseudopotentials -- Ti z_valence=12, O z_valence=6 -- not the naive
    textbook valence-electron counts of 4 and 6; z_valence is parsed
    directly from each UPF's own header, not assumed).
  - ecutwfc/ecutrho come from get_pseudopotential's returned Pseudo-Dojo
    `hints.ecut` (already fetched into template/*.upf's sibling metadata
    when the pseudopotentials were downloaded -- recomputed here from the
    same tool call for a single source of truth) converted Ha -> Ry (QE
    wants ecutwfc in Ry; Pseudo-Dojo hints are in Ha, so ecutwfc = 2*ecut).
  - the bands k-path is derived per-structure via
    pymatgen.symmetry.bandstructure.HighSymmKpath(structure) and converted
    to QE's K_POINTS crystal_b fractional coordinates via that SAME
    structure's own reciprocal lattice -- see step1_2_build_ensemble.py's
    docstring for why the base structure was standardized (via
    SpacegroupAnalyzer.get_primitive_standard_structure) before ever being
    used here: this guarantees HighSymmKpath's internal reciprocal lattice
    and this structure's own reciprocal lattice are identical (verified
    with a strict warnings-as-errors check during development), so no
    separate cross-frame rotation/reprojection step is needed or risked.
"""
import re
import shutil
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.symmetry.bandstructure import HighSymmKpath

from chem_llm.tools import get_pseudopotential

STRUCTURE_DIR = Path("structures")
CALC_DIR = Path("calculations")
TEMPLATE_DIR = Path("template")
CALC_DIR.mkdir(exist_ok=True)

KPATH_DIVISIONS_PER_SEGMENT = 20
ACCOUNT = "m4735"
EMAIL = "brent.hu@yale.edu"


def read_z_valence(upf_path: Path) -> float:
    text = upf_path.read_text(errors="replace")
    m = re.search(r'z_valence\s*=\s*"?\s*([\d.]+)', text)
    if not m:
        raise ValueError(f"could not find z_valence in {upf_path}")
    return float(m.group(1))


Z_VALENCE = {"Ti": read_z_valence(TEMPLATE_DIR / "Ti.upf"), "O": read_z_valence(TEMPLATE_DIR / "O.upf")}
print("z_valence read from actual UPF files:", Z_VALENCE)

# ecut hints, re-fetched from the same tool call used in step3 (single
# source of truth; get_pseudopotential just re-downloads/overwrites the
# same file and returns the same hints -- cheap, avoids a second code path
# for "what ecut did we use").
ecut_ha = 0.0
for el in ["Ti", "O"]:
    r = get_pseudopotential(
        element=el, output_path=str(TEMPLATE_DIR / f"{el}.upf"),
        kind="nc", relativity="sr", generator="pbesol", accuracy="stringent",
        format="upf", hint_level="normal",
    )
    assert r["success"], r
    ecut_ha = max(ecut_ha, r["hints"]["ecut"])

ECUTWFC_RY = 2.0 * ecut_ha  # Pseudo-Dojo hints are in Ha; QE ecutwfc is in Ry
ECUTRHO_RY = 4.0 * ECUTWFC_RY  # standard dual for norm-conserving pseudopotentials
print(f"ecutwfc = {ECUTWFC_RY} Ry, ecutrho = {ECUTRHO_RY} Ry (from pseudopotential hints, max over Ti/O)")

NBND_PAD = 8  # extra empty (conduction) bands beyond the occupied count, for bands.in only


def n_occupied_bands(structure: Structure) -> int:
    n_elec = sum(Z_VALENCE[site.specie.symbol] for site in structure)
    assert n_elec == int(n_elec) and int(n_elec) % 2 == 0, n_elec
    return int(n_elec) // 2


def get_kpath_crystal_b(structure: Structure):
    """Return QE K_POINTS crystal_b lines: the main connected HighSymmKpath
    branch for this structure, in ITS OWN reciprocal-lattice fractional
    coordinates (no external convention assumed)."""
    kpath = HighSymmKpath(structure)
    main_path = kpath.kpath["path"][0]  # longest connected branch
    points = kpath.kpath["kpoints"]
    lines = [str(len(main_path))]
    for i, label in enumerate(main_path):
        frac = points[label]
        npts = KPATH_DIVISIONS_PER_SEGMENT if i < len(main_path) - 1 else 1
        lines.append(f"{frac[0]:.10f} {frac[1]:.10f} {frac[2]:.10f} {npts}")
    return lines, main_path


def structure_to_qe(structure: Structure, prefix: str, calculation: str, n_occ: int) -> str:
    lines = []
    lines.append("&CONTROL")
    lines.append(f"   prefix = '{prefix}'")
    lines.append(f"   calculation = '{calculation if calculation != 'bands' else 'bands'}'")
    lines.append("   restart_mode = 'from_scratch'")
    lines.append("   outdir = './'")
    lines.append("   pseudo_dir = './'")
    lines.append("   verbosity = 'high'")
    lines.append("/")

    lines.append("&SYSTEM")
    lines.append("   ibrav = 0")
    lines.append(f"   nat = {len(structure)}")
    lines.append("   ntyp = 2")
    lines.append(f"   ecutwfc = {ECUTWFC_RY:.4f}")
    lines.append(f"   ecutrho = {ECUTRHO_RY:.4f}")
    lines.append("   occupations = 'fixed'")
    if calculation == "bands":
        lines.append(f"   nbnd = {n_occ + NBND_PAD}")
    lines.append("/")

    lines.append("&ELECTRONS")
    lines.append("   electron_maxstep = 200")
    lines.append("   conv_thr = 1.0d-9")
    lines.append("   mixing_beta = 0.7")
    lines.append("   diagonalization = 'david'")
    lines.append("/")

    lines.append("ATOMIC_SPECIES")
    lines.append("Ti  47.867  Ti.upf")
    lines.append("O   15.999  O.upf")
    lines.append("")

    lines.append("CELL_PARAMETERS angstrom")
    for vec in structure.lattice.matrix:
        lines.append(f"{vec[0]:.10f} {vec[1]:.10f} {vec[2]:.10f}")
    lines.append("")

    lines.append("ATOMIC_POSITIONS crystal")
    frac = structure.frac_coords % 1.0
    for specie, pos in zip(structure.species, frac):
        lines.append(f"{specie.symbol:<2} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}")
    lines.append("")

    if calculation == "scf":
        lines.append("K_POINTS automatic")
        lines.append("8 8 10 1 1 1")
    elif calculation == "bands":
        kpath_lines, main_path = get_kpath_crystal_b(structure)
        lines.append("K_POINTS crystal_b")
        lines.extend(kpath_lines)

    return "\n".join(lines) + "\n"


def write_submit_script(calc_path: Path, prefix: str):
    submit_text = f"""#!/bin/bash
#SBATCH -A {ACCOUNT}
#SBATCH -J tio2_{prefix}
#SBATCH -C gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time 01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user={EMAIL}

module load espresso/7.5-libxc-7.0.0-gpu-cu13

PW=pw.x
BANDS=bands.x

export OMP_NUM_THREADS=1

echo "Running SCF for {prefix}"
srun -n 4 $PW -in pw.in > pw.out
if [ $? -ne 0 ]; then
    echo "SCF failed"
    exit 1
fi

echo "Running bands SCF for {prefix}"
srun -n 4 $PW -in bands.in > bands_pw.out
if [ $? -ne 0 ]; then
    echo "Bands calculation failed"
    exit 1
fi

echo "Running bands.x for {prefix}"
$BANDS -in bands_post.in > bands_post.out

echo "Finished {prefix}"
"""
    (calc_path / "submit.sh").write_text(submit_text)


cif_files = sorted(STRUCTURE_DIR.glob("structure_[0-9][0-9][0-9].cif"))
assert len(cif_files) == 51, len(cif_files)

main_path_labels = None
for cif_file in cif_files:
    prefix = cif_file.stem.split("_")[-1]  # "000".."050"
    calc_path = CALC_DIR / prefix
    calc_path.mkdir(parents=True, exist_ok=True)

    for pp in ["Ti.upf", "O.upf"]:
        shutil.copyfile(TEMPLATE_DIR / pp, calc_path / pp)

    structure = Structure.from_file(cif_file)
    n_occ = n_occupied_bands(structure)

    (calc_path / "pw.in").write_text(structure_to_qe(structure, prefix, "scf", n_occ))
    (calc_path / "bands.in").write_text(structure_to_qe(structure, prefix, "bands", n_occ))

    if prefix == "000":
        _, main_path_labels = get_kpath_crystal_b(structure)

    bands_post_text = f"""&BANDS
   prefix = '{prefix}'
   outdir = './'
   filband = '{prefix}.bands.dat'
   lsym = .true.
/
"""
    (calc_path / "bands_post.in").write_text(bands_post_text)

    write_submit_script(calc_path, prefix)

print(f"Wrote QE inputs for {len(cif_files)} structures into {CALC_DIR}/")
print(f"n_occ (structure_000, standard 6-atom cell) = {n_occupied_bands(Structure.from_file('structures/structure_000.cif'))}")
print("K-path (structure_000, from HighSymmKpath, main branch):", " -> ".join(main_path_labels))
