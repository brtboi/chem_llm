"""Scraper for the Quantum ESPRESSO pw.x input documentation
(https://www.quantum-espresso.org/Doc/INPUT_PW.html).

The page is hand-written HTML, not Sphinx output, with its own recurring
pattern: every namelist/card variable is introduced by
`<a name="VARNAME"></a><table>...</table>`, where the table's first row
is the variable name + type, the following row(s) are labelled fields
(Status/Default/See ...), and the last row holds the free-text
description in a <blockquote><pre>. Section boundaries (which namelist or
card a variable belongs to) are given by preceding `<h2>` headings
("Namelist: &SYSTEM", "Card: K_POINTS ..."), so the page is walked once,
top to bottom, tracking the most recent heading.
"""
import logging
import re

import requests
from bs4 import BeautifulSoup, Tag

from .. import config
from ..models import Document

logger = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": config.HTTP_USER_AGENT})

_SECTION_RE = re.compile(r"^(Namelist|Card)\s*:\s*&?\s*([A-Za-z0-9_]+)")


def _get(url: str) -> str:
    resp = _SESSION.get(url, timeout=config.HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def _parse_section_heading(h2: Tag) -> tuple[str, str] | None:
    m = _SECTION_RE.match(h2.get_text(" ", strip=True))
    if not m:
        return None
    kind, name = m.groups()
    return kind.lower(), name


def _variable_name(table: Tag) -> str | None:
    anchor = table.find_previous("a", attrs={"name": True})
    if anchor is None:
        return None
    return anchor["name"]


def _parse_variable_table(table: Tag, var_name: str, section_type: str, section_name: str) -> Document | None:
    rows = table.find_all("tr", recursive=False)
    if not rows:
        return None

    header_cells = rows[0].find_all(["th", "td"], recursive=False)
    if len(header_cells) < 2:
        return None
    var_type = header_cells[1].get_text(" ", strip=True)

    fields: dict[str, str] = {}
    description = ""
    for row in rows[1:]:
        cells = row.find_all("td", recursive=False)
        if len(cells) == 2 and not cells[0].find("blockquote"):
            label = cells[0].get_text(" ", strip=True).rstrip(":")
            value = cells[1].get_text(" ", strip=True)
            if label:
                fields[label] = value
        else:
            blockquote = row.find("blockquote")
            target = blockquote if blockquote else row
            description = target.get_text("\n", strip=True)

    lines = [f"Namelist &{section_name}" if section_type == "namelist" else f"Card {section_name}"]
    lines.append(f"Variable: {var_name} ({var_type})")
    for label, value in fields.items():
        lines.append(f"{label}: {value}")
    if description:
        lines.append("")
        lines.append(description)

    metadata = {
        "source": "quantum_espresso",
        "section_type": section_type,
        "section": section_name,
        "variable": var_name,
        "type": var_type,
        **{k.lower(): v for k, v in fields.items()},
    }

    return Document(
        text="\n".join(lines),
        title=f"{section_name} / {var_name}",
        url=f"{config.QE_PW_DOC_URL}#{var_name}",
        source="quantum_espresso",
        metadata=metadata,
    )


def scrape_quantum_espresso(url: str | None = None) -> list[Document]:
    """Scrape the pw.x INPUT_PW documentation into one Document per
    namelist/card input variable.
    """
    url = url or config.QE_PW_DOC_URL

    try:
        html = _get(url)
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return []

    soup = BeautifulSoup(html, "html.parser")

    documents: list[Document] = []
    section_type, section_name = "namelist", "UNKNOWN"
    seen_vars: set[str] = set()

    for el in soup.body.find_all(["h2", "table"]):
        if el.name == "h2":
            parsed = _parse_section_heading(el)
            if parsed:
                section_type, section_name = parsed
            continue

        # el.name == "table": only tables preceded directly by a named
        # anchor and shaped like a variable block are variable defs; the
        # page also has plain layout tables which this filters out.
        var_name = _variable_name(el)
        if not var_name or var_name in seen_vars or var_name.startswith("id"):
            continue
        rows = el.find_all("tr", recursive=False)
        if not rows or not rows[0].find("th", recursive=False):
            continue

        doc = _parse_variable_table(el, var_name, section_type, section_name)
        if doc is not None:
            documents.append(doc)
            seen_vars.add(var_name)

    logger.info("quantum_espresso: %s -> %d variables", url, len(documents))
    return documents
