"""Structure-aware chunking.

Each Document coming out of the scrapers is already one semantic unit --
one pymatgen class/function/method/property/attribute, or one QE
namelist/card variable -- so the *primary* chunk boundary is the document
boundary itself, not a fixed character window. We only split a document
further when its own text exceeds chunk_size, in which case we split on
paragraph (then sentence) boundaries with overlap, never mid-identifier.

Every chunk is prefixed with a short header carrying the document's
identifying fields (qualified_name for pymatgen, section/variable for QE)
so a chunk read in isolation -- e.g. the 3rd chunk of a long class
docstring -- never loses the "what is this" context that made it
findable in the first place.
"""
import re

from retrieval import config
from retrieval.models import Chunk, Document

_PARA_SPLIT_RE = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _header_for(doc: Document) -> str:
    meta = doc.metadata
    if doc.source == "pymatgen":
        return f"[{meta.get('qualified_name', doc.title)}]"
    if doc.source == "quantum_espresso":
        section = meta.get("section", "")
        variable = meta.get("variable", doc.title)
        return f"[QE {meta.get('section_type', 'section')} {section} / {variable}]"
    return f"[{doc.title}]"


def _split_long_unit(unit: str, size: int) -> list[str]:
    """Split a single paragraph/sentence-group that's still over `size`
    chars, on sentence boundaries, falling back to a hard split only if a
    single sentence itself exceeds `size`.
    """
    sentences = _SENTENCE_SPLIT_RE.split(unit)
    pieces, current = [], ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= size or not current:
            current = candidate
        else:
            pieces.append(current)
            current = sentence
        while len(current) > size:
            pieces.append(current[:size])
            current = current[size:]
    if current:
        pieces.append(current)
    return pieces


def _pack_paragraphs(paragraphs: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current = ""

    def flush():
        nonlocal current
        if current:
            chunks.append(current.strip())
        current = ""

    for para in paragraphs:
        if len(para) > size:
            for piece in _split_long_unit(para, size):
                candidate = f"{current}\n\n{piece}".strip() if current else piece
                if len(candidate) <= size or not current:
                    current = candidate
                else:
                    flush()
                    current = piece
            continue

        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= size or not current:
            current = candidate
        else:
            flush()
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{para}".strip() if tail else para

    flush()
    return chunks or [""]


def chunk_document(
    doc: Document,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    header = _header_for(doc)
    body = doc.text.strip()

    if len(header) + len(body) + 2 <= chunk_size:
        pieces = [body]
    else:
        paragraphs = [p.strip() for p in _PARA_SPLIT_RE.split(body) if p.strip()]
        pieces = _pack_paragraphs(paragraphs, size=max(chunk_size - len(header) - 2, 200), overlap=overlap)

    doc_key = doc.metadata.get("qualified_name") or doc.metadata.get("variable") or doc.url

    chunks = []
    for i, piece in enumerate(pieces):
        text = f"{header}\n{piece}" if piece else header
        chunk_metadata = dict(doc.metadata)
        chunk_metadata["chunk_index"] = i
        chunk_metadata["num_chunks"] = len(pieces)
        chunks.append(
            Chunk(
                id=f"{doc.source}:{doc_key}:{i}",
                text=text,
                title=doc.title,
                url=doc.url,
                source=doc.source,
                metadata=chunk_metadata,
            )
        )
    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))
    return chunks
