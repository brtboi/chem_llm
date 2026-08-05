"""Scraper for the pymatgen API documentation (https://pymatgen.org/).

pymatgen's docs are Sphinx-autodoc pages, one per top-level subpackage
(pymatgen.core.html, pymatgen.io.html, ...). Each page is a single large
HTML document containing every class/function/method/property/attribute
in that subpackage as a `<dl class="py ...">` block, with the fully
qualified dotted name as the block's anchor id. Methods/properties/
attributes are nested inside their owning class's block.

We deliberately do NOT flatten the page to plain text: identifiers
(qualified_name, class, method/function name) are pulled from the anchor
id and kept as structured metadata on every Document, because exact API
lookups ("Structure.from_file") depend on that identifier surviving
chunking/indexing verbatim -- see chunking.py, which also prefixes each
chunk's text with these identifiers.
"""
import logging
import re

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

from retrieval import config
from retrieval.models import Document

logger = logging.getLogger(__name__)

# dl.py.<KIND> classes emitted by Sphinx's Python domain for API members.
_MEMBER_KINDS = {"class", "function", "method", "property", "attribute", "exception", "data"}

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": config.HTTP_USER_AGENT})


def _get(url: str) -> str:
    resp = _SESSION.get(url, timeout=config.HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def discover_module_pages() -> list[str]:
    """Crawl the pymatgen API index page for every `pymatgen.*.html`
    module page it links to. Falls back to the static list in config.py
    if the index can't be reached, so a build never hard-fails just
    because the index page layout changed.
    """
    try:
        html = _get(config.PYMATGEN_INDEX_URL)
    except Exception as e:
        logger.warning("Could not fetch pymatgen index (%s); using fallback module list", e)
        return [f"{m}.html" for m in config.PYMATGEN_MODULES]

    soup = BeautifulSoup(html, "html.parser")
    pages = sorted(
        {
            a["href"]
            for a in soup.find_all("a", href=True)
            if a["href"].startswith("pymatgen") and a["href"].endswith(".html")
        }
    )
    return pages or [f"{m}.html" for m in config.PYMATGEN_MODULES]


def _render_text(node: Tag) -> str:
    """Render a docstring-body node to plain text, keeping <pre> code
    blocks as fenced, multi-line code instead of collapsing them to one
    space-joined line.
    """
    node = BeautifulSoup(str(node), "html.parser")
    for pre in node.find_all("pre"):
        code = pre.get_text("\n")
        pre.replace_with(NavigableString(f"\n```\n{code.strip()}\n```\n"))
    return re.sub(r"[ \t]+", " ", node.get_text(" ", strip=True))


def _own_text(dd: Tag) -> str:
    """Text belonging to this member's own docstring/description, i.e.
    every direct child of <dd> that is NOT a nested `dl.py.*` block
    (those are separate, nested members -- see module docstring).
    """
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
        "source": "pymatgen",
        "module": module,
        "qualified_name": qualified_name,
        "kind": kind,
        "name": bare_name,
        "class": class_name,
    }
    # Also expose a same-named key (e.g. metadata["method"] = "from_file")
    # to match the literal {"class": ..., "method": ...} shape requested
    # for exact-field lookups, in addition to the generalized kind/name.
    metadata[kind] = bare_name

    return Document(
        text=text,
        title=qualified_name,
        url=f"{page_url}#{qualified_name}",
        source="pymatgen",
        metadata=metadata,
    )


def _parse_module_page(page: str) -> list[Document]:
    page_url = config.PYMATGEN_BASE_URL + page
    module = page[: -len(".html")]

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


def scrape_pymatgen(modules: list[str] | None = None) -> list[Document]:
    """Scrape pymatgen API docs into Documents, one per class/function/
    method/property/attribute/exception/data member.

    `modules` may be a list of module page filenames (e.g.
    ["pymatgen.core.html"]) to restrict the crawl; defaults to every page
    discovered via discover_module_pages().
    """
    pages = modules if modules is not None else discover_module_pages()

    documents: list[Document] = []
    for page in pages:
        page_docs = _parse_module_page(page)
        logger.info("pymatgen: %s -> %d members", page, len(page_docs))
        documents.extend(page_docs)

    return documents
