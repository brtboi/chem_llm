#!/usr/bin/env python3
"""Regenerate chem_llm/tools/qe_namelists.py from the QE helpdoc pages.

    python scripts/build_qe_namelists.py

Scrapes config.QE_DOC_URLS (pw.x + bands.x) with the same scraper
search_docs uses, and writes out {variable: NAMELIST} per binary. The
result is committed so validate_qe_input can check namelist membership
without a built retrieval index or network access.
"""
import collections
from pathlib import Path

from chem_llm import REPO_ROOT
from chem_llm.retrieval.scrapers.quantum_espresso import scrape_quantum_espresso

OUT = REPO_ROOT / "chem_llm" / "tools" / "qe_namelists.py"


def main():
    per = collections.defaultdict(dict)
    for doc in scrape_quantum_espresso():
        meta = doc.metadata
        if meta.get("section_type") != "namelist":
            continue
        binary = "bands" if "INPUT_BANDS" in doc.url else "pw"
        per[binary][meta["variable"].lower()] = meta["section"].upper()

    lines = [
        '"""Which Quantum ESPRESSO namelist each input variable belongs to.',
        "",
        "Generated from the scraped QE helpdoc pages (config.QE_DOC_URLS) --",
        "the same source search_docs indexes -- and committed so",
        "validate_qe_input works without a built retrieval index.",
        "",
        "Regenerate with scripts/build_qe_namelists.py after a QE doc update.",
        '"""',
        "",
    ]
    for binary in ("pw", "bands"):
        lines.append(f"{binary.upper()}_VARIABLE_NAMELIST = {{")
        for var in sorted(per[binary]):
            lines.append(f"    {var!r}: {per[binary][var]!r},")
        lines.append("}")
        lines.append("")
    lines.append('VARIABLE_NAMELIST = {"pw": PW_VARIABLE_NAMELIST, "bands": BANDS_VARIABLE_NAMELIST}')

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}: " + ", ".join(f"{b}={len(d)} variables" for b, d in per.items()))


if __name__ == "__main__":
    main()
