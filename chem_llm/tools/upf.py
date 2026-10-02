"""Read the facts a DFT setup needs from a UPF pseudopotential file.

Everything here comes from the file's own PP_HEADER, not from a lookup table:
z_valence (electrons the pseudopotential treats as valence -- what QE sums
into 'number of electrons'), whether it is scalar- or fully-relativistic,
its type and functional. UPF v2 stores these as attributes of one
<PP_HEADER .../> element; the older v1 text block is handled as a fallback.

Note the generation-input block at the top of an ONCVPSP file
("Cs 55.00 9 3 ..." = symbol, Z, core states, VALENCE STATES) counts
orbitals, not electrons -- read z_valence, never that line.
"""
from __future__ import annotations

import re
from pathlib import Path

__all__ = ["read_upf_header"]

_V2_HEADER = re.compile(r"<PP_HEADER\b(.*?)/?>", re.S)
_ATTR = re.compile(r'([A-Za-z_]+)\s*=\s*"([^"]*)"')


def _bool(value: str) -> bool:
    return value.strip().strip(".").upper() in {"T", "TRUE"}


def read_upf_header(path: str | Path) -> dict:
    """Header facts of a UPF file. Raises ValueError if there is no header."""
    text = Path(path).read_text(errors="replace")
    v2 = re.search(r'<UPF version="([^"]+)"', text)
    m = _V2_HEADER.search(text)

    if v2 and m:
        a = {k: v.strip() for k, v in _ATTR.findall(m.group(1))}
        relativistic = a.get("relativistic", "").lower()
        facts = {
            "upf_version": v2.group(1),
            "element": a.get("element"),
            "z_valence": float(a["z_valence"]),
            "pseudo_type": a.get("pseudo_type"),
            "relativistic": relativistic,
            "spin_orbit": _bool(a.get("has_so", "F")) or relativistic == "full",
            "functional": a.get("functional"),
            "core_correction": _bool(a.get("core_correction", "F")),
            "l_max": int(a["l_max"]) if a.get("l_max", "").lstrip("-").isdigit() else None,
        }
        # The header's wfc_cutoff/rho_cutoff are deliberately not reported: in
        # ONCVPSP files they are not plane-wave cutoffs (e.g. rho_cutoff=15.85
        # for a Ti that needs ~84 Ry), so they would mislead. Recommended
        # cutoffs come from the pseudo-dojo hints instead.
        return facts

    block = re.search(r"<PP_HEADER>(.*?)</PP_HEADER>", text, re.S)
    if not block:
        raise ValueError(f"{path}: no PP_HEADER found -- not a UPF file?")
    lines = block.group(1).splitlines()
    find = lambda label: next((l.split()[0] for l in lines if label.lower() in l.lower()), None)
    z = find("Z valence")
    if z is None:
        raise ValueError(f"{path}: UPF v1 header without a 'Z valence' line")
    return {
        "upf_version": "1",
        "element": find("Element"),
        "z_valence": float(z.replace("D", "E").replace("d", "e")),
        "pseudo_type": find("US|NC|PAW") or find("Ultrasoft") or None,
        "relativistic": None,  # v1 headers do not record it reliably
        "spin_orbit": None,
        "functional": None,
        "core_correction": _bool(find("Nonlinear Core Correction") or "F"),
        "l_max": None,
    }
