#!/usr/bin/env python3
"""Build (or rebuild) the documentation retrieval index.

    python scripts/build_doc_index.py
    python scripts/build_doc_index.py --sources pymatgen
    python scripts/build_doc_index.py --no-vector   # BM25 only, fast/offline

A thin CLI over chem_llm.retrieval.build.build_index, which is the same
call DocumentationIndex.build() makes. The index lands in --index-dir,
defaulting to paths.doc_index_dir from the config.yaml found from cwd.

Requires chem_llm to be installed (`uv sync`, from the repo root).
"""
import argparse
import json
import logging
import sys

from chem_llm.retrieval import config
from chem_llm.retrieval.build import SCRAPERS, build_index
from chem_llm.settings import Settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_doc_index")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources", nargs="+", choices=list(SCRAPERS), default=list(SCRAPERS),
        help="Which documentation sources to scrape (default: all).",
    )
    parser.add_argument("--index-dir", default=None,
                        help="Where to write the index (default: doc_index_dir from config.yaml).")
    parser.add_argument("--no-vector", action="store_true", help="Skip building the vector index.")
    parser.add_argument("--no-bm25", action="store_true", help="Skip building the BM25 index.")
    parser.add_argument("--embedding-model", default=None, help="Override config.EMBEDDING_MODEL.")
    parser.add_argument(
        "--chunk-size", type=int, default=config.CHUNK_SIZE, help="Soft target chunk size, in characters.",
    )
    parser.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    args = parser.parse_args()

    settings = Settings.discover()
    try:
        build_info = build_index(
            args.sources,
            index_dir=args.index_dir or settings.doc_index_dir,
            cache_dir=settings.hf_cache_dir,
            embed=not args.no_vector,
            bm25=not args.no_bm25,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)

    logger.info("Done. %s", json.dumps(build_info))


if __name__ == "__main__":
    main()
