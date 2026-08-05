"""
Scraper for official pymatgen documentation.

Provides:
    scan_pymatgen_docs()

Uses pymatgen's generated API pages directly instead of
Sphinx inventory/search files.
"""

import difflib
import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup


_BASE = "https://pymatgen.org/"
_API_INDEX = _BASE + "pymatgen.html"

_CACHE_FILE = Path.home() / ".cache" / "pymatgen_docs_index.json"

_INDEX_CACHE = None


def _build_index():
    """
    Crawl pymatgen API pages and build:

    {
        "structure": [
            {
                "name": "...Structure",
                "page": "...html",
                "anchor": "..."
            }
        ]
    }
    """

    index = {}

    resp = requests.get(_API_INDEX, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    pages = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]

        if (
            href.endswith(".html")
            and href.startswith("pymatgen")
        ):
            pages.add(href)


    for page in pages:

        resp = requests.get(
            _BASE + page,
            timeout=30,
        )

        resp.raise_for_status()

        soup = BeautifulSoup(
            resp.text,
            "html.parser",
        )

        for dl in soup.find_all("dl"):

            dt = dl.find("dt")

            if dt is None:
                continue

            anchor = dt.get("id")

            if not anchor:
                continue

            name = dt.get_text(
                " ",
                strip=True,
            )

            # remove signatures
            bare = (
                name.split("(")[0]
                .split(".")[-1]
                .strip()
            )

            if not bare:
                continue

            key = bare.lower()

            index.setdefault(
                key,
                [],
            ).append(
                {
                    "name": name,
                    "page": page,
                    "anchor": anchor,
                }
            )


    return index


def _load_index():

    global _INDEX_CACHE

    if _INDEX_CACHE is not None:
        return _INDEX_CACHE


    if _CACHE_FILE.exists():

        try:
            _INDEX_CACHE = json.loads(
                _CACHE_FILE.read_text()
            )

            return _INDEX_CACHE

        except Exception:
            pass


    _INDEX_CACHE = _build_index()

    _CACHE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    _CACHE_FILE.write_text(
        json.dumps(
            _INDEX_CACHE,
            indent=2,
        )
    )

    return _INDEX_CACHE



def _extract_doc(page, anchor):

    resp = requests.get(
        _BASE + page,
        timeout=30,
    )

    resp.raise_for_status()

    soup = BeautifulSoup(
        resp.text,
        "html.parser",
    )


    node = soup.find(
        id=anchor
    )

    if node is None:
        return {}


    container = node

    while (
        container
        and container.name != "dl"
    ):
        container = container.parent


    if container is None:
        return {}


    dt = container.find("dt")
    dd = container.find("dd")


    return {
        "signature": (
            dt.get_text(
                " ",
                strip=True,
            )
            if dt else ""
        ),

        "docstring": (
            dd.get_text(
                "\n",
                strip=True,
            )
            if dd else ""
        ),

        "url": (
            _BASE
            + page
            + "#"
            + anchor
        ),
    }



def scan_pymatgen_docs(
    query: str,
    class_hint: str | None = None,
    max_results: int = 5,
):

    try:
        index = _load_index()

    except Exception as e:
        return {
            "success": False,
            "stderr": str(e),
        }


    if "." in query:

        cls, _, member = query.rpartition(".")

        key = member.lower()

        hint = class_hint or cls

    else:

        key = query.lower()

        hint = class_hint


    results = index.get(
        key,
        []
    )


    if hint:

        hint = hint.lower()

        results = [
            r
            for r in results
            if hint in r["name"].lower()
        ]


    if not results:

        return {
            "success": False,
            "stderr": (
                f"No pymatgen entry found "
                f"matching '{query}'"
            ),

            "suggestions": difflib.get_close_matches(
                key,
                index.keys(),
                n=5,
            ),
        }


    output = []

    for r in results[:max_results]:

        doc = _extract_doc(
            r["page"],
            r["anchor"],
        )

        if doc:

            doc["matched_as"] = r["name"]
            output.append(doc)


    return {
        "success": True,
        "query": query,
        "num_matches": len(results),
        "results": output,
    }