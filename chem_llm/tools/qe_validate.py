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

from . import register_tool

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


@register_tool(
    "validate_qe_input",
    (
        "Parse a Quantum ESPRESSO pw.x input file and check internal "
        "consistency: that &SYSTEM's nat equals the number of atoms listed "
        "under ATOMIC_POSITIONS, and that ntyp equals the number of species "
        "listed under ATOMIC_SPECIES. Catches the common bug where nat/ntyp "
        "is hardcoded or copied from an earlier note instead of computed "
        "from the actual structure object, causing it to silently drift out "
        "of sync with the real atom count. This is a deterministic file "
        "check, not a correctness judgement -- it cannot tell you whether "
        "nat is the *scientifically* right number, only whether it agrees "
        "with the cards in the same file. Call this on every pw.in-style "
        "file (SCF and bands inputs) right after writing it, and again after "
        "any edit to it, before run_espresso_workflow or done."
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

    issues = []
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

    return {
        "success": not issues,
        "path": path,
        "nat_declared": declared.get("nat"),
        "num_atomic_positions": len(positions),
        "ntyp_declared": declared.get("ntyp"),
        "num_atomic_species": len(species),
        "issues": issues,
    }
