"""Pretrained sentence-embedding wrapper. No training happens here --
`EmbeddingModel` only ever calls `.encode()` on an off-the-shelf
sentence-transformers model, which model is entirely configurable via
retrieval/config.py (EMBEDDING_MODEL) or the DOC_EMBEDDING_MODEL env var.

Loading the model is expensive (downloads + moves weights to device), so
it's deferred until the first `embed_*` call rather than done at import
or __init__ time -- this keeps `import retrieval.embeddings` cheap for
callers (like tools/docs.py) that may not need it on every process.
"""
import numpy as np

from retrieval import config


class EmbeddingModel:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed a batch of chunk texts at indexing time. Returns an
        (N, dim) float32 array, L2-normalized so downstream similarity
        is a plain dot product (see vector_store.py).
        """
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 200,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query at search time. Kept separate from
        embed_documents so it's obvious/enforced at the call site that
        only the query (never the corpus) is embedded during a search --
        document embeddings are computed once, at build time.
        """
        return self.embed_documents([text])[0]
