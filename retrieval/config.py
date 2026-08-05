"""Central configuration for the documentation retrieval system.

Everything a build/query run needs to know -- which models to use, how big
chunks are, how many candidates to pull at each stage, where the index
lives on disk, and which URLs to scrape -- lives here so nothing else in
retrieval/ hardcodes it. All paths are resolved to absolute paths at import
time (mirroring config.py's LOG_FILE pattern) so behavior doesn't change if
the caller later os.chdir()s, which main.py does.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Models (pretrained only -- nothing here is trained/fine-tuned) ---
# Both are swappable via env var without touching code. Defaults are small
# CPU-friendly sentence-transformers models so the index can be built and
# queried without a GPU; swap for larger models if quality matters more
# than latency/memory.
EMBEDDING_MODEL = os.environ.get(
    "DOC_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
RERANKER_MODEL = os.environ.get(
    "DOC_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# --- Chunking ---
# Character-based (not token-based) to keep chunking dependency-free;
# CHUNK_SIZE is a soft target, not a hard cap -- see chunking.py.
CHUNK_SIZE = int(os.environ.get("DOC_CHUNK_SIZE", 900))
CHUNK_OVERLAP = int(os.environ.get("DOC_CHUNK_OVERLAP", 150))

# --- Retrieval ---
VECTOR_TOP_K = int(os.environ.get("DOC_VECTOR_TOP_K", 25))
BM25_TOP_K = int(os.environ.get("DOC_BM25_TOP_K", 25))
FUSED_TOP_K = int(os.environ.get("DOC_FUSED_TOP_K", 30))  # candidates handed to the reranker
FINAL_TOP_K = int(os.environ.get("DOC_FINAL_TOP_K", 5))   # results returned by search_docs
RRF_K = 60  # standard reciprocal-rank-fusion damping constant
# Cap on how many chunks from the *same* source document may appear in the
# final results, so a single long entry (e.g. a QE variable with many
# enumerated options, split across several chunks) can't crowd out
# otherwise-relevant hits from other documents.
MAX_CHUNKS_PER_DOCUMENT = int(os.environ.get("DOC_MAX_CHUNKS_PER_DOCUMENT", 2))

# --- Index location ---
INDEX_DIR = Path(os.environ.get("DOC_INDEX_DIR", REPO_ROOT / "retrieval_index")).resolve()
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
VECTOR_INDEX_DIR = INDEX_DIR / "vector"
BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"
BUILD_INFO_PATH = INDEX_DIR / "build_info.json"

# --- Documentation sources ---
PYMATGEN_BASE_URL = "https://pymatgen.org/"
PYMATGEN_INDEX_URL = PYMATGEN_BASE_URL + "pymatgen.html"
# Fallback list used only if the index page can't be crawled (e.g. offline
# build). Kept in sync with the pages linked from pymatgen.html as of
# pymatgen 2026.x; the scraper's normal path discovers this list live.
PYMATGEN_MODULES = [
    "pymatgen.core",
    "pymatgen.io",
    "pymatgen.analysis",
    "pymatgen.electronic_structure",
    "pymatgen.symmetry",
    "pymatgen.transformations",
    "pymatgen.entries",
    "pymatgen.phonon",
    "pymatgen.ext",
    "pymatgen.command_line",
    "pymatgen.util",
    "pymatgen.vis",
    "pymatgen.apps",
    "pymatgen.alchemy",
    "pymatgen.cli",
    "pymatgen.optimization",
]

QE_PW_DOC_URL = "https://www.quantum-espresso.org/Doc/INPUT_PW.html"

HTTP_TIMEOUT = 30
HTTP_USER_AGENT = "chem_llm-doc-retrieval/1.0 (+https://github.com)"
