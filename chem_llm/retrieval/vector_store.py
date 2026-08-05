"""Flat in-memory vector index with cosine (dot-product-of-normalized-
vectors) similarity, persisted as a .npy matrix + a JSON id list.

A full ANN library (faiss, hnswlib, ...) is deliberately not used here:
the corpus is a few thousand documentation chunks, for which a single
(N, dim) x (dim,) matrix-vector product is sub-millisecond and exact --
no approximation, no extra native-code dependency to install on an HPC
cluster. If the corpus grows by an order of magnitude or more, swapping
this module for a faiss-backed one is a self-contained change; nothing
else in retrieval/ depends on how similarity search is implemented
internally.
"""
import json
from pathlib import Path

import numpy as np

from . import config


class VectorStore:
    def __init__(self):
        self.ids: list[str] = []
        self.vectors: np.ndarray | None = None  # (N, dim), L2-normalized

    def build(self, ids: list[str], vectors: np.ndarray) -> None:
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids and vectors must have the same length")
        self.ids = list(ids)
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if self.vectors is None or len(self.ids) == 0:
            return []
        scores = self.vectors @ np.asarray(query_vector, dtype=np.float32)
        top_k = min(top_k, len(self.ids))
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self.ids[i], float(scores[i])) for i in top_idx]

    def save(self, path: Path | None = None) -> None:
        path = path or config.VECTOR_INDEX_DIR
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        (path / "ids.json").write_text(json.dumps(self.ids))

    @classmethod
    def load(cls, path: Path | None = None) -> "VectorStore":
        path = path or config.VECTOR_INDEX_DIR
        store = cls()
        store.vectors = np.load(path / "vectors.npy")
        store.ids = json.loads((path / "ids.json").read_text())
        return store

    @classmethod
    def exists(cls, path: Path | None = None) -> bool:
        path = path or config.VECTOR_INDEX_DIR
        return (path / "vectors.npy").exists() and (path / "ids.json").exists()
