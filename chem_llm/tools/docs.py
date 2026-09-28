"""Agent tool wrapping the hybrid documentation retrieval system in
chem_llm/retrieval/. Registers `search_docs` against the same
TOOLS/TOOL_DISPATCH registry as every other tool in tools/__init__.py.

Which index is searched comes from the running agent's settings
(doc_index_dir), so two agents pointed at different indexes each search
their own. A retriever is built once per index directory and reused --
constructing one only loads the index files (cheap); the embedding and
reranker models load on its first search, not at import time, so importing
tools doesn't require a GPU or network access.
"""
from pathlib import Path

from ..retrieval import config as retrieval_config
from ..retrieval.hybrid_search import DocRetriever
from ..settings import active_settings
from . import register_tool

_retrievers: dict[tuple[Path, Path | None], DocRetriever] = {}


def get_retriever(index_dir: Path, cache_dir: Path | None = None) -> DocRetriever:
    """The shared retriever for one index directory (built on first use).
    Raises FileNotFoundError if no index has been built there."""
    key = (Path(index_dir).resolve(), cache_dir)
    if key not in _retrievers:
        _retrievers[key] = DocRetriever(index_dir=key[0], cache_dir=cache_dir)
    return _retrievers[key]


@register_tool(
    "search_docs",
    (
        "Search indexed technical documentation (ASE -- Atomic Simulation "
        "Environment -- API reference, including ase.io.espresso's "
        "write_espresso_in/read_espresso_out, and Quantum ESPRESSO pw.x and "
        "bands.x input variables) for text relevant to a natural-language question or an "
        "exact API identifier. Combines keyword (BM25) and semantic "
        "(embedding) search, then reranks with a cross-encoder. Does not "
        "call an LLM -- results are retrieved documentation chunks, not "
        "generated answers. Use this before guessing ASE class/method "
        "signatures or Quantum ESPRESSO namelist variable names/units. "
        "Examples of good queries: 'How do I read a CIF file with ASE?', "
        "'ecutwfc', 'How do I set smearing parameters?', 'Atoms.get_scaled_positions'."
    ),
    {
        "query": "string. Natural-language question or exact identifier to search for.",
        "top_k": (
            "int (optional). Number of results to return "
            f"(default {retrieval_config.FINAL_TOP_K})."
        ),
        "sources": (
            "list[string] (optional). Restrict results to these sources: "
            "'ase', 'quantum_espresso'. Default: search both."
        ),
    },
)
def search_docs(query: str, top_k: int | None = None, sources: list[str] | None = None):
    settings = active_settings()
    try:
        retriever = get_retriever(settings.doc_index_dir, settings.hf_cache_dir)
    except FileNotFoundError as e:
        return {"success": False, "stderr": str(e)}

    results = retriever.search(
        query,
        top_k=top_k or retrieval_config.FINAL_TOP_K,
        sources=sources,
    )

    return {
        "success": True,
        "query": query,
        "num_results": len(results),
        "results": [r.to_dict() for r in results],
    }
