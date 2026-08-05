#!/usr/bin/env python3
"""Example usage of the documentation retrieval system.

Run after building the index (see scripts/build_doc_index.py). Requires
chem_llm to be installed (`uv sync`, from the repo root):

    python scripts/example_search_docs.py
"""
import json

EXAMPLE_QUERIES = [
    "How do I create a pymatgen Structure from a CIF?",
    "What does ecutwfc control in QE?",
    "How do I set smearing parameters?",
    "What class handles electronic structure calculations?",
    "How do I configure k-points?",
    "Structure.from_file",
]


def via_agent_tool():
    """The way the agent actually calls this: through tools.TOOL_DISPATCH,
    exactly as agent_core.execute_tool() would for a `search_docs` tool
    call emitted by the model.
    """
    from chem_llm.tools import TOOL_DISPATCH

    print("=== via the search_docs agent tool ===\n")
    for query in EXAMPLE_QUERIES:
        result = TOOL_DISPATCH["search_docs"](query, top_k=3)
        print(f"query: {query!r}")
        if not result["success"]:
            print(f"  ERROR: {result['stderr']}")
            continue
        for hit in result["results"]:
            print(f"  [{hit['score']:.3f}] ({hit['source']}) {hit['title']}")
            print(f"      {hit['url']}")
        print()


def via_retriever_directly():
    """Lower-level usage, e.g. for evaluation/debugging outside the agent
    loop -- construct a DocRetriever once and reuse it.
    """
    from chem_llm.retrieval.hybrid_search import DocRetriever

    print("=== via chem_llm.retrieval.hybrid_search.DocRetriever directly ===\n")
    retriever = DocRetriever()

    query = "How do I set smearing parameters?"
    results = retriever.search(query, top_k=3, sources=["quantum_espresso"])
    print(f"query: {query!r} (restricted to quantum_espresso)")
    for r in results:
        print(json.dumps(r.to_dict(), indent=2)[:500])
        print("---")


if __name__ == "__main__":
    via_agent_tool()
    via_retriever_directly()
