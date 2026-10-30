"""The structures Materials Project has for one formula, one per space group.

Used by generate_cif when no space group is given (the agent then picks the
phase the task names, e.g. "rutile") and by the guided setup's search.
"""
from __future__ import annotations

MP_FIELDS = ["material_id", "symmetry", "energy_above_hull", "band_gap", "nsites", "theoretical"]


def group_by_spacegroup(docs) -> list[dict]:
    """Materials Project summary docs -> one entry per space group: its most
    stable structure (the one generate_cif fetches for that space group),
    `observed` if any entry in the space group matches an experimental (ICSD)
    structure. Observed space groups first, then by energy above the hull --
    by PBE energy alone, hypothetical carbon allotropes rank above graphite
    and diamond."""
    best: dict[int, dict] = {}
    observed: set[int] = set()
    for d in docs:
        sym = getattr(d, "symmetry", None)
        if sym is None or sym.number is None:
            continue
        entry = {
            "spacegroup_symbol": sym.symbol,
            "spacegroup_number": int(sym.number),
            "crystal_system": str(sym.crystal_system or "").split(".")[-1].lower(),
            "material_id": str(d.material_id),
            "nsites": getattr(d, "nsites", None),
            "energy_above_hull_eV_per_atom": d.energy_above_hull,
            "band_gap_eV": d.band_gap,
        }
        if getattr(d, "theoretical", None) is False:
            observed.add(entry["spacegroup_number"])
        old = best.get(entry["spacegroup_number"])
        e_new, e_old = entry["energy_above_hull_eV_per_atom"], old and old["energy_above_hull_eV_per_atom"]
        if old is None or (e_new if e_new is not None else 1e9) < (e_old if e_old is not None else 1e9):
            best[entry["spacegroup_number"]] = entry
    for e in best.values():
        e["observed"] = e["spacegroup_number"] in observed
    return sorted(best.values(), key=lambda e: (not e["observed"], e["energy_above_hull_eV_per_atom"] is None,
                                                e["energy_above_hull_eV_per_atom"] or 0))


def lookup_polymorphs(formula: str, api_key: str | None) -> list[dict]:
    from mp_api.client import MPRester

    with MPRester(api_key, mute_progress_bars=True) as mpr:
        docs = mpr.materials.summary.search(formula=formula, fields=MP_FIELDS)
    return group_by_spacegroup(docs)


def parse_spacegroup(text: str) -> tuple[str, int]:
    """'P4_2/mnm' or '136' -> ('P4_2/mnm', 136). Raises ValueError."""
    from pymatgen.symmetry.groups import SpaceGroup

    text = text.strip()
    if text.isdigit():
        n = int(text)
        if not 1 <= n <= 230:
            raise ValueError("space group numbers run from 1 to 230")
        return SpaceGroup.from_int_number(n).symbol, n
    try:
        n = SpaceGroup(text).int_number
    except Exception:
        raise ValueError(f"{text!r} is not a space group symbol pymatgen recognizes "
                         "(Materials Project format: P4_2/mnm, Fd-3m, P6_3/mmc)") from None
    # Materials Project's spelling, whatever was typed (P42/mnm -> P4_2/mnm)
    return SpaceGroup.from_int_number(n).symbol, n


def classify_spacegroup(text: str) -> tuple[str | None, int | None, str]:
    """What the user typed for the structure -> (symbol, number, phase name).

    '136' and 'P4_2/mnm' are space groups; anything else that is not a valid
    symbol ('rutile', 'zinc blende') is a phase name, which the agent
    resolves against Materials Project's structures for the formula. A typo
    in a symbol ('P4_2/mmn') therefore becomes a name -- callers say so."""
    text = text.strip()
    try:
        symbol, number = parse_spacegroup(text)
        return symbol, number, ""
    except ValueError:
        if text.isdigit():
            raise
        return None, None, text
