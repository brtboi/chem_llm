"""Hybrid retrieval: vector search + BM25, fused with Reciprocal Rank
Fusion, then (optionally) reranked with a cross-encoder.

query
  |
  +--> vector_store.search(embed_query(query))  --\
  |                                                 +--> RRF fusion --> dedup --> candidates
  +--> bm25_index.search(query)                  --/
                                                        |
                                                        v
                                                    reranker.rerank()
                                                        |
                                                        v
                                                  top-k ScoredChunk

RRF (score = sum of 1/(k + rank) across the lists a chunk appears in) is
used instead of normalizing and summing raw scores because BM25 scores
and cosine similarities live on incomparable scales; RRF only needs each
list's *ranking*, not its scores, to combine them, which is what makes it
the standard choice for this kind of fusion.
"""
import json
import logging

from retrieval import config
from retrieval.bm25 import BM25Index
from retrieval.embeddings import EmbeddingModel
from retrieval.models import Chunk, ScoredChunk
from retrieval.reranker import Reranker
from retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _load_chunks(path=None) -> dict[str, Chunk]:
    path = path or config.CHUNKS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No chunk store found at {path}. Run scripts/build_doc_index.py first."
        )
    chunks = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = Chunk.from_dict(json.loads(line))
            chunks[chunk.id] = chunk
    return chunks


def _doc_key(chunk: Chunk) -> str:
    """Identifies the *parent document* a chunk came from, so results can
    be diversified across documents rather than across chunks.
    """
    meta = chunk.metadata
    return meta.get("qualified_name") or meta.get("variable") or chunk.url


def _diversify(scored: list[ScoredChunk], top_k: int, max_per_document: int) -> list[ScoredChunk]:
    """Keep results in their given (score-descending) order, but stop
    admitting chunks from a document once `max_per_document` of its
    chunks are already in the result -- otherwise a single long entry
    split into several chunks (e.g. a QE variable with many enumerated
    options) can dominate the top-k at the expense of other documents.
    """
    counts: dict[str, int] = {}
    out = []
    for sc in scored:
        key = _doc_key(sc.chunk)
        if counts.get(key, 0) >= max_per_document:
            continue
        counts[key] = counts.get(key, 0) + 1
        out.append(sc)
        if len(out) >= top_k:
            break
    return out


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]], k: int = config.RRF_K
) -> dict[str, float]:
    """ranked_lists: list of [(chunk_id, raw_score), ...] already sorted
    best-first. Returns {chunk_id: fused_score}, higher is better.
    """
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (chunk_id, _raw_score) in enumerate(ranked):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


class DocRetriever:
    """Loads the persisted vector + BM25 indices and chunk store once,
    and answers `search()` calls against them. Embedding/reranker models
    are loaded lazily on first use (see EmbeddingModel/Reranker), so
    constructing a DocRetriever is cheap even before any query is made.
    """

    def __init__(
        self,
        index_dir=None,
        embedding_model: str | None = None,
        reranker_model: str | None = None,
    ):
        index_dir = index_dir or config.INDEX_DIR
        chunks_path = index_dir / "chunks.jsonl"
        vector_path = index_dir / "vector"
        bm25_path = index_dir / "bm25.pkl"

        self.chunks = _load_chunks(chunks_path)
        self.vector_store = VectorStore.load(vector_path) if VectorStore.exists(vector_path) else None
        self.bm25_index = BM25Index.load(bm25_path) if BM25Index.exists(bm25_path) else None

        if self.vector_store is None and self.bm25_index is None:
            raise FileNotFoundError(
                f"No vector or BM25 index found under {index_dir}. Run scripts/build_doc_index.py first."
            )

        self._embedder = EmbeddingModel(embedding_model)
        self._reranker = Reranker(reranker_model)

    def search(
        self,
        query: str,
        top_k: int = config.FINAL_TOP_K,
        vector_top_k: int = config.VECTOR_TOP_K,
        bm25_top_k: int = config.BM25_TOP_K,
        fused_top_k: int = config.FUSED_TOP_K,
        rerank: bool = True,
        sources: list[str] | None = None,
        max_chunks_per_document: int = config.MAX_CHUNKS_PER_DOCUMENT,
    ) -> list[ScoredChunk]:
        vector_hits: list[tuple[str, float]] = []
        if self.vector_store is not None:
            query_vector = self._embedder.embed_query(query)
            vector_hits = self.vector_store.search(query_vector, vector_top_k)

        bm25_hits: list[tuple[str, float]] = []
        if self.bm25_index is not None:
            bm25_hits = self.bm25_index.search(query, bm25_top_k)

        vector_scores = dict(vector_hits)
        bm25_scores = dict(bm25_hits)
        fused_scores = reciprocal_rank_fusion([vector_hits, bm25_hits])

        candidate_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)[:fused_top_k]

        candidates = []
        for chunk_id in candidate_ids:
            chunk = self.chunks.get(chunk_id)
            if chunk is None:
                continue  # stale index entry; chunk store and index out of sync
            if sources and chunk.source not in sources:
                continue
            candidates.append(
                ScoredChunk(
                    chunk=chunk,
                    score=fused_scores[chunk_id],
                    vector_score=vector_scores.get(chunk_id),
                    bm25_score=bm25_scores.get(chunk_id),
                )
            )

        if rerank and candidates:
            reranked = self._reranker.rerank(query, candidates, top_k=len(candidates))
            return _diversify(reranked, top_k, max_chunks_per_document)

        candidates.sort(key=lambda c: c.score, reverse=True)
        return _diversify(candidates, top_k, max_chunks_per_document)
