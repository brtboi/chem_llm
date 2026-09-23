"""Scraper for the ASE (Atomic Simulation Environment) API documentation
(https://docs.ase-lib.org/).

Like pymatgen's docs (see scrapers/pymatgen.py), ASE's docs are Sphinx-
autodoc pages with every class/function/method/property/attribute as a
`<dl class="py ...">` block, anchored by its fully qualified dotted name --
so the same block-parsing logic applies verbatim. The one structural
difference is page layout: pymatgen has one page per top-level subpackage
(`pymatgen.core.html`), while ASE's pages are nested by sub-module
(`ase/atoms.html`, `ase/dft/kpoints.html`, `ase/io/io.html`, ...), so
discovery here crawls `py-modindex.html` (Sphinx's standard module index)
filtered to a curated allowlist of pages relevant to this pipeline
(structure building/IO, k-paths, symmetry) instead of pulling in ASE's
full page set (dozens of third-party calculator backends we never call).
"""
import logging
import re

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

from .. import config
from ..models import Document

logger = logging.getLogger(__name__)

_MEMBER_KINDS = {"class", "function", "method", "property", "attribute", "exception", "data"}

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": config.HTTP_USER_AGENT})


def _get(url: str) -> str:
    resp = _SESSION.get(url, timeout=config.HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def discover_module_pages() -> list[str]:
    """Crawl ASE's `py-modindex.html` for module page paths, restricted to
    `config.ASE_PAGE_ALLOWLIST` (this pipeline only ever touches a small
    slice of ASE -- structure building, CIF/QE IO, k-paths, symmetry --
    not the dozens of third-party calculator backends ASE also documents).
    Falls back to the static allowlist itself if the index can't be
    reached, so a build never hard-fails just because the page changed.
    """
    try:
        html = _get(config.ASE_MODINDEX_URL)
    except Exception as e:
        logger.warning("Could not fetch ASE module index (%s); using allowlist as-is", e)
        return list(config.ASE_PAGE_ALLOWLIST)

    soup = BeautifulSoup(html, "html.parser")
    available = {
        a["href"] for a in soup.find_all("a", href=True) if a["href"].endswith(".html")
    }
    pages = [p for p in config.ASE_PAGE_ALLOWLIST if p in available]
    missing = set(config.ASE_PAGE_ALLOWLIST) - set(pages)
    if missing:
        logger.warning("ASE doc pages in allowlist but not found in module index: %s", sorted(missing))
    return pages or list(config.ASE_PAGE_ALLOWLIST)


def _render_text(node: Tag) -> str:
    node = BeautifulSoup(str(node), "html.parser")
    for pre in node.find_all("pre"):
        code = pre.get_text("\n")
        pre.replace_with(NavigableString(f"\n```\n{code.strip()}\n```\n"))
    return re.sub(r"[ \t]+", " ", node.get_text(" ", strip=True))


def _own_text(dd: Tag) -> str:
    parts = []
    for child in dd.find_all(recursive=False):
        if child.name == "dl" and child.get("class") and "py" in child.get("class"):
            continue
        text = _render_text(child)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _parse_member(dl: Tag, module: str, page_url: str) -> Document | None:
    dt = dl.find("dt", recursive=False) or dl.find("dt")
    if dt is None or not dt.get("id"):
        return None

    kind = next((c for c in (dl.get("class") or []) if c in _MEMBER_KINDS), None)
    if kind is None:
        return None

    qualified_name = dt["id"]
    bare_name = qualified_name.rsplit(".", 1)[-1]
    parent_path = qualified_name.rsplit(".", 1)[0] if "." in qualified_name else ""
    class_name = parent_path.rsplit(".", 1)[-1] if kind in ("method", "property", "attribute") else (
        bare_name if kind == "class" else None
    )

    signature = dt.get_text(" ", strip=True)
    signature = re.sub(r"\s*\[source\]\s*$", "", signature)

    dd = dl.find("dd", recursive=False)
    description = _own_text(dd) if dd else ""

    text = f"{kind} {qualified_name}\n\n{signature}"
    if description:
        text += f"\n\n{description}"

    metadata = {
        "source": "ase",
        "module": module,
        "qualified_name": qualified_name,
        "kind": kind,
        "name": bare_name,
        "class": class_name,
    }
    metadata[kind] = bare_name

    return Document(
        text=text,
        title=qualified_name,
        url=f"{page_url}#{qualified_name}",
        source="ase",
        metadata=metadata,
    )


def _parse_module_page(page: str) -> list[Document]:
    page_url = config.ASE_BASE_URL + page
    module = page[: -len(".html")].replace("/", ".")

    try:
        html = _get(page_url)
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", page_url, e)
        return []

    soup = BeautifulSoup(html, "html.parser")

    docs: list[Document] = []
    for dl in soup.find_all("dl"):
        classes = dl.get("class") or []
        if "py" not in classes:
            continue
        doc = _parse_member(dl, module=module, page_url=page_url)
        if doc is not None:
            docs.append(doc)

    return docs


def scrape_ase(modules: list[str] | None = None) -> list[Document]:
    """Scrape ASE API docs into Documents, one per class/function/method/
    property/attribute/exception/data member.

    `modules` may be a list of page paths (e.g. ["ase/atoms.html"]) to
    restrict the crawl; defaults to `discover_module_pages()`.
    """
    pages = modules if modules is not None else discover_module_pages()

    documents: list[Document] = []
    for page in pages:
        page_docs = _parse_module_page(page)
        logger.info("ase: %s -> %d members", page, len(page_docs))
        documents.extend(page_docs)

    return documents
