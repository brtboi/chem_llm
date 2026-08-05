"""Pretrained cross-encoder reranker. Like embeddings.py, this only ever
calls `.predict()` on an off-the-shelf sentence-transformers CrossEncoder
-- no training -- and the model is swappable via config.RERANKER_MODEL /
DOC_RERANKER_MODEL. Loading is deferred to first use for the same reason
as EmbeddingModel.
"""
from retrieval import config
from retrieval.models import ScoredChunk


class Reranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.RERANKER_MODEL
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        model = self._load()
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = model.predict(pairs)

        reranked = []
        for candidate, score in zip(candidates, scores):
            reranked.append(
                ScoredChunk(
                    chunk=candidate.chunk,
                    score=float(score),
                    vector_score=candidate.vector_score,
                    bm25_score=candidate.bm25_score,
                    rerank_score=float(score),
                )
            )
        reranked.sort(key=lambda c: c.score, reverse=True)
        return reranked[:top_k]
