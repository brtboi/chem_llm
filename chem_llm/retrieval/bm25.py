"""BM25 keyword index, for exact-identifier queries (`Structure.from_file`,
`ecutwfc`, `K_POINTS`, `MPRelaxSet`) that dense embeddings are unreliable
at matching verbatim. Backed by rank_bm25's plain-Python BM25Okapi --
adequate for a documentation-sized corpus and has no native build step.

The tokenizer is the one BM25-specific design choice worth calling out:
identifiers are indexed both whole ("structure.from_file",
"k_points") and split on "." / "_" ("structure", "from_file", "k",
"points"), so a query can hit either the fully-qualified form or a bare
member/variable name.
"""
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from . import config
from .models import Chunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for m in _TOKEN_RE.finditer(text.lower()):
        tok = m.group(0)
        tokens.append(tok)
        if "." in tok:
            tokens.extend(part for part in tok.split(".") if part)
        if "_" in tok:
            tokens.extend(part for part in tok.split("_") if part)
    return tokens


class BM25Index:
    def __init__(self):
        self.ids: list[str] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self.ids = [c.id for c in chunks]
        corpus = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._bm25 is None or not self.ids:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self.ids)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.ids[i], float(scores[i])) for i in ranked if scores[i] > 0]

    def save(self, path: Path | None = None) -> None:
        path = path or config.BM25_INDEX_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"ids": self.ids, "bm25": self._bm25}, f)

    @classmethod
    def load(cls, path: Path | None = None) -> "BM25Index":
        path = path or config.BM25_INDEX_PATH
        index = cls()
        with path.open("rb") as f:
            data = pickle.load(f)
        index.ids = data["ids"]
        index._bm25 = data["bm25"]
        return index

    @classmethod
    def exists(cls, path: Path | None = None) -> bool:
        path = path or config.BM25_INDEX_PATH
        return path.exists()
