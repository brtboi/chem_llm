"""Pretrained cross-encoder reranker. Like embeddings.py, this only ever
calls `.predict()` on an off-the-shelf sentence-transformers CrossEncoder
-- no training -- and the model is swappable via config.RERANKER_MODEL /
DOC_RERANKER_MODEL. Loading is deferred to first use for the same reason
as EmbeddingModel.
"""
from . import config
from .models import ScoredChunk


class Reranker:
    def __init__(self, model_name: str | None = None, cache_dir=None):
        self.model_name = model_name or config.RERANKER_MODEL
        self.cache_dir = cache_dir  # hub cache; None -> library default / HF_HOME
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            # sentence-transformers 5.x: CrossEncoder forwards cache_folder
            # to its Transformer module via an argument that module itself
            # deprecated, so it logs a deprecation warning at us -- then
            # routes the directory correctly. Nothing for us to change;
            # silence that one logger for the duration of the load.
            import logging

            decorators_log = logging.getLogger("sentence_transformers.util.decorators")
            level = decorators_log.level
            decorators_log.setLevel(logging.ERROR)
            try:
                self._model = CrossEncoder(
                    self.model_name, cache_folder=str(self.cache_dir) if self.cache_dir else None
                )
            finally:
                decorators_log.setLevel(level)
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
