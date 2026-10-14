"""Agent tool for catching a specific, recurring class of bug: a Quantum
ESPRESSO pw.x input file whose &SYSTEM nat/ntyp were hardcoded (typed from
memory, or copied from an earlier `note`) instead of computed from the
actual structure object, so they silently drift out of sync with the real
ATOMIC_POSITIONS/ATOMIC_SPECIES cards. pw.x either errors confusingly on
this or -- worse -- silently truncates/misreads atoms, so it needs to be
caught before a run, not diagnosed after one.

This is a deterministic parser, not an LLM call: it does not know whether
`nat` is scientifically correct, only whether it is internally consistent
with the cards in the same file.
"""
import re
from pathlib import Path

from ..kpath_check import check_bands_kpath
from . import register_tool
from .qe_namelists import VARIABLE_NAMELIST

# Card names that terminate whichever card is currently being scanned.
_CARD_NAMES = {
    "atomic_species",
    "atomic_positions",
    "cell_parameters",
    "k_points",
    "constraints",
    "occupations",
    "atomic_velocities",
    "atomic_forces",
    "add_output_atomic_forces",
    "hubbard",
    "solvents",
}

_NAMELIST_INT_RE = re.compile(r"(?im)^\s*(nat|ntyp)\s*=\s*(\d+)")

_NAMELIST_OPEN_RE = re.compile(r"^\s*&\s*([A-Za-z_]+)")
_ASSIGNMENT_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s*=")


def _detect_binary(text: str) -> str:
    """'bands' for a bands.x input (&BANDS), else 'pw'."""
    return "bands" if re.search(r"(?im)^\s*&\s*bands\b", text) else "pw"


def _check_namelist_membership(text: str, binary: str) -> list[str]:
    """Flag variables written into the wrong namelist.

    pw.x reports this as a parse error naming the namelist it was reading
    ("bad line in namelist &control: conv_thr = 1e-08"), which reads like a
    formatting complaint about the value -- so it is easy to spend many
    edits reformatting a number that was never the problem. The variable
    just belongs in a different namelist.
    """
    known = VARIABLE_NAMELIST[binary]

    issues: list[str] = []
    current: str | None = None
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        opened = _NAMELIST_OPEN_RE.match(stripped)
        if opened:
            current = opened.group(1).upper()
            continue
        if stripped.startswith("/"):
            current = None
            continue
        if current is None:
            continue  # inside a card, not a namelist
        assigned = _ASSIGNMENT_RE.match(stripped)
        if not assigned:
            continue
        var = assigned.group(1).lower()
        expected = known.get(var)
        if expected is None:
            issues.append(
                f"'{var}' in &{current} is not a documented {binary}.x input variable "
                "(check spelling, or whether it belongs to a different executable)"
            )
        elif expected != current:
            issues.append(
                f"'{var}' is in &{current} but belongs in &{expected} -- move it "
                f"(pw.x will report this as 'bad line in namelist &{current.lower()}', "
                "which is about placement, not number formatting)"
            )
    return issues


def _extract_card_lines(text: str, card: str) -> list[str]:
    """Return the non-empty data lines belonging to `card` (e.g.
    'atomic_positions'), stopping at the next card/namelist or a blank line
    following data already collected."""
    lines: list[str] = []
    in_card = False
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if in_card and lines:
                break
            continue
        first_word = stripped.split()[0].lower()
        if first_word == card:
            in_card = True
            continue
        if in_card:
            if first_word in _CARD_NAMES or stripped.startswith("&"):
                break
            lines.append(stripped)
    return lines


def _check_spin_orbit(text: str) -> list[str]:
    """Elements past Xe need spin-orbit (the relativistic DECISION RULE);
    sweep 8's CsPbBr3 ran without it (gap 1.60 eV instead of ~0.39)."""
    from ase.data import atomic_numbers

    species = re.findall(r"^\s*([A-Z][a-z]?)\d*\s+[-+0-9.eE]+\s+\S+\.(?:upf|UPF)\b", text, re.M)
    heavy = sorted({s for s in species if atomic_numbers.get(s, 0) > 54})
    if heavy and not re.search(r"^\s*lspinorb\s*=\s*\.?t", text, re.M | re.I):
        return [f"{', '.join(heavy)} (atomic number > 54) without lspinorb = .true.: the relativistic DECISION "
                "RULE requires fully relativistic pseudopotentials and spin-orbit (noncolin and lspinorb) "
                "for any element past Xe, whatever the task targets"]
    return []


def _check_magnetization(text: str) -> list[str]:
    """A spin-polarized input whose starting magnetizations are all zero
    converges to the non-magnetic state with no error (sweep 4's Fe:
    starting_magnetization = 0.5 in input_data, overridden by the
    starting_magnetization(1) = 0.0 ASE writes from the atoms' moments)."""
    if not re.search(r"^\s*nspin\s*=\s*2\b", text, re.M | re.I):
        return []
    issues = []
    if re.search(r"^\s*starting_magnetization\s*=", text, re.M | re.I):
        issues.append("starting_magnetization without a species index: pw.x needs starting_magnetization(i) "
                      "per species. With write_espresso_in, set the moments on the Atoms "
                      "(atoms.set_initial_magnetic_moments) instead of in input_data -- ASE writes "
                      "starting_magnetization(i) from them")
    values = [float(v.replace("d", "e").replace("D", "e")) for v in re.findall(
        r"^\s*starting_magnetization\(\d+\)\s*=\s*([-+0-9.eEdD]+)", text, re.M | re.I)]
    if not values or all(abs(v) < 1e-8 for v in values):
        issues.append("nspin = 2 but every starting_magnetization(i) is zero (or missing): the SCF will converge to "
                      "the non-magnetic state. Set the initial moments on the cell, e.g. "
                      "prim.set_initial_magnetic_moments([...]) after kp_atoms(kp), before write_espresso_in")
    return issues


@register_tool(
    "validate_qe_input",
    (
        "Parse a Quantum ESPRESSO input file (pw.x or bands.x) and check it "
        "two ways. (1) Internal consistency: &SYSTEM's nat equals the number "
        "of atoms under ATOMIC_POSITIONS, and ntyp equals the number of "
        "species under ATOMIC_SPECIES -- catching the bug where nat/ntyp is "
        "hardcoded or copied from an earlier note instead of computed from "
        "the actual structure object. (2) Namelist membership: every "
        "variable is a documented input variable of that executable AND is "
        "in the namelist the docs put it in -- catching e.g. conv_thr "
        "written into &CONTROL when it belongs in &ELECTRONS, which pw.x "
        "reports as 'bad line in namelist &control: conv_thr = ...', an "
        "error that reads like a complaint about the number's format but is "
        "really about where the variable sits. This is a deterministic file "
        "check, not a correctness judgement -- it cannot tell you whether "
        "nat is the *scientifically* right number, only whether it agrees "
        "with the cards in the same file. (3) For a bands input "
        "(calculation='bands'): every segment endpoint of the k-path is a real "
        "high-symmetry point (a missing segment start after a discontinuity "
        "shows up as an endpoint that is not one); if a kpath.json from "
        "chem_llm.kpath sits next to it, that the cell and every k-point are "
        "exactly the ones in it. Call this on every pw.in-style file (SCF and "
        "bands inputs) right after writing it, "
        "and again after any edit to it, before run_espresso_workflow or done."
    ),
    {"path": "string. Path to the QE input file to validate."},
)
def validate_qe_input(path: str):
    file_path = Path(path)
    if not file_path.exists():
        return {"success": False, "stderr": f"{path} does not exist"}

    text = file_path.read_text(errors="replace")

    declared = {m.group(1).lower(): int(m.group(2)) for m in _NAMELIST_INT_RE.finditer(text)}
    positions = _extract_card_lines(text, "atomic_positions")
    species = _extract_card_lines(text, "atomic_species")

    binary = _detect_binary(text)

    issues = []
    if binary == "bands":
        # bands.x input: a single &BANDS namelist, no structure cards, so the
        # nat/ntyp consistency check below does not apply.
        issues.extend(_check_namelist_membership(text, binary))
        return {
            "success": not issues,
            "path": path,
            "binary": "bands.x",
            "issues": issues,
        }

    if "nat" not in declared:
        issues.append("no 'nat' found in &SYSTEM")
    elif declared["nat"] != len(positions):
        issues.append(
            f"nat={declared['nat']} but ATOMIC_POSITIONS lists {len(positions)} atoms"
        )
    if "ntyp" not in declared:
        issues.append("no 'ntyp' found in &SYSTEM")
    elif declared["ntyp"] != len(species):
        issues.append(
            f"ntyp={declared['ntyp']} but ATOMIC_SPECIES lists {len(species)} species"
        )

    issues.extend(_check_namelist_membership(text, binary))
    issues.extend(check_bands_kpath(file_path, file_path.parent / "band_labels.dat"))
    issues.extend(_check_magnetization(text))
    issues.extend(_check_spin_orbit(text))

    return {
        "success": not issues,
        "path": path,
        "binary": "pw.x",
        "nat_declared": declared.get("nat"),
        "num_atomic_positions": len(positions),
        "ntyp_declared": declared.get("ntyp"),
        "num_atomic_species": len(species),
        "issues": issues,
    }
