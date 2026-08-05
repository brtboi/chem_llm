#!/usr/bin/env python3
"""Build (or rebuild) the documentation retrieval index.

    python scripts/build_doc_index.py
    python scripts/build_doc_index.py --sources pymatgen
    python scripts/build_doc_index.py --no-vector   # BM25 only, fast/offline

Pipeline: scrape -> chunk -> write chunks.jsonl -> embed chunks -> save
vector index -> build+save BM25 index. Document embeddings are computed
here, once; search-time code (chem_llm/retrieval/hybrid_search.py) only
ever embeds the query.

Requires chem_llm to be installed (`uv sync`, from the repo root).
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

from chem_llm.retrieval import config
from chem_llm.retrieval.bm25 import BM25Index
from chem_llm.retrieval.chunking import chunk_documents
from chem_llm.retrieval.embeddings import EmbeddingModel
from chem_llm.retrieval.models import Chunk, Document
from chem_llm.retrieval.scrapers.pymatgen import scrape_pymatgen
from chem_llm.retrieval.scrapers.quantum_espresso import scrape_quantum_espresso
from chem_llm.retrieval.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_doc_index")

SCRAPERS = {
    "pymatgen": scrape_pymatgen,
    "quantum_espresso": scrape_quantum_espresso,
}


def scrape(sources: list[str]) -> list[Document]:
    documents: list[Document] = []
    for source in sources:
        t0 = time.perf_counter()
        docs = SCRAPERS[source]()
        logger.info("%s: scraped %d documents in %.1fs", source, len(docs), time.perf_counter() - t0)
        documents.extend(docs)
    return documents


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict()) + "\n")


def build_vector_index(chunks: list[Chunk], embedding_model: str | None) -> None:
    embedder = EmbeddingModel(embedding_model)
    texts = [c.text for c in chunks]
    logger.info("Embedding %d chunks with %s ...", len(texts), embedder.model_name)
    vectors = embedder.embed_documents(texts)

    store = VectorStore()
    store.build(ids=[c.id for c in chunks], vectors=vectors)
    store.save()
    logger.info("Saved vector index (%d x %d) to %s", *vectors.shape, config.VECTOR_INDEX_DIR)


def build_bm25_index(chunks: list[Chunk]) -> None:
    index = BM25Index()
    index.build(chunks)
    index.save()
    logger.info("Saved BM25 index (%d chunks) to %s", len(chunks), config.BM25_INDEX_PATH)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources", nargs="+", choices=list(SCRAPERS), default=list(SCRAPERS),
        help="Which documentation sources to scrape (default: all).",
    )
    parser.add_argument("--no-vector", action="store_true", help="Skip building the vector index.")
    parser.add_argument("--no-bm25", action="store_true", help="Skip building the BM25 index.")
    parser.add_argument("--embedding-model", default=None, help="Override config.EMBEDDING_MODEL.")
    parser.add_argument(
        "--chunk-size", type=int, default=config.CHUNK_SIZE, help="Soft target chunk size, in characters.",
    )
    parser.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    args = parser.parse_args()

    documents = scrape(args.sources)
    if not documents:
        logger.error("No documents scraped -- aborting without touching the existing index.")
        sys.exit(1)

    chunks = chunk_documents(documents, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
    logger.info("Chunked %d documents into %d chunks", len(documents), len(chunks))

    write_chunks(chunks, config.CHUNKS_PATH)
    logger.info("Wrote chunk store to %s", config.CHUNKS_PATH)

    if not args.no_vector:
        build_vector_index(chunks, args.embedding_model)
    if not args.no_bm25:
        build_bm25_index(chunks)

    build_info = {
        "sources": args.sources,
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "embedding_model": args.embedding_model or config.EMBEDDING_MODEL,
        "built_vector_index": not args.no_vector,
        "built_bm25_index": not args.no_bm25,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    config.BUILD_INFO_PATH.write_text(json.dumps(build_info, indent=2))
    logger.info("Done. %s", json.dumps(build_info))


if __name__ == "__main__":
    main()
