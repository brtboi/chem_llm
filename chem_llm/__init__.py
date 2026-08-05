from pathlib import Path

# The repo root, one level above this package directory. Works the same
# under an editable install (uv sync) since __file__ still points at the
# real source location on disk, not a copied site-packages path. Every
# other path in this codebase (logs, the agent working directory, the
# doc-retrieval index) is anchored to this instead of cwd, so behavior
# doesn't depend on where a script/notebook happened to be launched from.
REPO_ROOT = Path(__file__).resolve().parent.parent
