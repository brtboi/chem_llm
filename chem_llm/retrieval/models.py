"""Common data model shared by every scraper, chunker, and index in
retrieval/. Scrapers only ever produce Document objects; everything
downstream (chunking, indexing, search) only ever consumes/produces
Chunk and ScoredChunk objects.
"""
from dataclasses import dataclass, field


@dataclass
class Document:
    """One scraped documentation page/entry, before chunking."""

    text: str
    title: str
    url: str
    source: str  # e.g. "pymatgen" or "quantum_espresso"
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrieval-sized unit of a Document. `id` is stable across
    rebuilds as long as the source document and chunk boundaries don't
    change, so it can be used as the join key between the vector store
    and the BM25 index.
    """

    id: str
    text: str
    title: str
    url: str
    source: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(
            id=d["id"],
            text=d["text"],
            title=d["title"],
            url=d["url"],
            source=d["source"],
            metadata=d.get("metadata", {}),
        )


@dataclass
class ScoredChunk:
    """A Chunk plus the score(s) it was retrieved with. The individual
    retriever scores are kept (not just the final one) so callers/agents
    can see *why* something was retrieved.
    """

    chunk: Chunk
    score: float
    vector_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.chunk.text,
            "title": self.chunk.title,
            "url": self.chunk.url,
            "source": self.chunk.source,
            "metadata": self.chunk.metadata,
            "score": self.score,
            "vector_score": self.vector_score,
            "bm25_score": self.bm25_score,
            "rerank_score": self.rerank_score,
        }
