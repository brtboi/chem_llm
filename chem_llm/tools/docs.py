"""Agent tool wrapping the hybrid documentation retrieval system in
chem_llm/retrieval/. Registers `search_docs` against the same
TOOLS/TOOL_DISPATCH registry as every other tool in tools/__init__.py.

The retriever (and the embedding/reranker models it lazily owns) is
built once per process on first call and reused after that -- loading a
DocRetriever is index-load-only (cheap); the actual models only load on
the first `search_docs` call, not at import time, so importing tools
doesn't require a GPU or network access.
"""
from ..retrieval import config as retrieval_config
from ..retrieval.hybrid_search import DocRetriever
from . import register_tool

_retriever: DocRetriever | None = None


def _get_retriever() -> DocRetriever:
    global _retriever
    if _retriever is None:
        _retriever = DocRetriever()
    return _retriever


@register_tool(
    "search_docs",
    (
        "Search indexed technical documentation (pymatgen API reference and "
        "Quantum ESPRESSO pw.x input variables) for text relevant to a "
        "natural-language question or an exact API identifier. Combines "
        "keyword (BM25) and semantic (embedding) search, then reranks with a "
        "cross-encoder. Does not call an LLM -- results are retrieved "
        "documentation chunks, not generated answers. Use this before "
        "guessing pymatgen class/method signatures or Quantum ESPRESSO "
        "namelist variable names/units. Examples of good queries: "
        "'How do I create a pymatgen Structure from a CIF?', 'ecutwfc', "
        "'How do I set smearing parameters?', 'Structure.from_file'."
    ),
    {
        "query": "string. Natural-language question or exact identifier to search for.",
        "top_k": (
            "int (optional). Number of results to return "
            f"(default {retrieval_config.FINAL_TOP_K})."
        ),
        "sources": (
            "list[string] (optional). Restrict results to these sources: "
            "'pymatgen', 'quantum_espresso'. Default: search both."
        ),
    },
)
def search_docs(query: str, top_k: int | None = None, sources: list[str] | None = None):
    try:
        retriever = _get_retriever()
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
