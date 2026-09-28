"""Build (or rebuild) the documentation index: scrape -> chunk -> embed ->
save.

One function, used by both entry points -- `scripts/build_doc_index.py` on
the command line and `DocumentationIndex.build()` in the package API -- so
an index built either way is the same index.

Document embeddings are computed here, once. Search-time code
(hybrid_search.DocRetriever) only ever embeds the query.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from . import config
from .bm25 import BM25Index
from .chunking import chunk_documents
from .models import Chunk, Document
from .scrapers.ase_docs import scrape_ase
from .scrapers.pymatgen import scrape_pymatgen
from .scrapers.quantum_espresso import scrape_quantum_espresso

logger = logging.getLogger(__name__)

SCRAPERS = {
    "ase": scrape_ase,
    "pymatgen": scrape_pymatgen,
    "quantum_espresso": scrape_quantum_espresso,
}

# pymatgen is scraped on request but not by default: the workflow the agent
# runs goes through ASE, and pymatgen's reference is large enough to dilute
# the hits for the pages that matter.
DEFAULT_SOURCES = ("ase", "quantum_espresso")


def scrape(sources) -> list[Document]:
    documents: list[Document] = []
    for source in sources:
        if source not in SCRAPERS:
            raise ValueError(f"Unknown documentation source {source!r}; expected one of {sorted(SCRAPERS)}")
        started = time.perf_counter()
        docs = SCRAPERS[source]()
        logger.info("%s: scraped %d documents in %.1fs", source, len(docs), time.perf_counter() - started)
        documents.extend(docs)
    return documents


def build_index(
    sources=DEFAULT_SOURCES,
    *,
    index_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    embed: bool = True,
    bm25: bool = True,
    embedding_model: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict:
    """Scrape `sources` and write a fresh index into `index_dir` (default
    config.INDEX_DIR). `cache_dir` is the Hugging Face hub cache the
    embedding model is downloaded into.

    Needs network access. Returns the build_info record, which is also
    written into the index directory.
    """
    index_dir = Path(index_dir or config.INDEX_DIR).resolve()
    sources = list(sources)
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = config.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap

    documents = scrape(sources)
    if not documents:
        # Never half-write over a working index because the network was down.
        raise RuntimeError(f"No documents scraped from {sources}; left the existing index untouched")

    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=chunk_overlap)
    logger.info("Chunked %d documents into %d chunks", len(documents), len(chunks))
    _write_chunks(chunks, index_dir / "chunks.jsonl")

    if embed:
        from .embeddings import EmbeddingModel
        from .vector_store import VectorStore

        embedder = EmbeddingModel(embedding_model, cache_dir=cache_dir)
        logger.info("Embedding %d chunks with %s ...", len(chunks), embedder.model_name)
        vectors = embedder.embed_documents([c.text for c in chunks])
        store = VectorStore()
        store.build(ids=[c.id for c in chunks], vectors=vectors)
        store.save(index_dir / "vector")
        logger.info("Saved vector index (%d x %d) to %s", *vectors.shape, index_dir / "vector")

    if bm25:
        index = BM25Index()
        index.build(chunks)
        index.save(index_dir / "bm25.pkl")
        logger.info("Saved BM25 index (%d chunks) to %s", len(chunks), index_dir / "bm25.pkl")

    build_info = {
        "index_dir": str(index_dir),
        "sources": sources,
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embedding_model or config.EMBEDDING_MODEL,
        "built_vector_index": embed,
        "built_bm25_index": bm25,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (index_dir / "build_info.json").write_text(json.dumps(build_info, indent=2))
    return build_info


def _write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict()) + "\n")
    logger.info("Wrote chunk store to %s", path)
