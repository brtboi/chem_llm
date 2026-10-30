"""Chat models already downloaded to the Hugging Face cache, and downloading one."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .settings import Settings

DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_MODEL_GB = 56  # bf16 weights, as downloaded

_ARCH_SUFFIXES = ("ForCausalLM", "ForConditionalGeneration")


def _snapshot(repo) -> Path | None:
    revs = sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True)
    return Path(revs[0].snapshot_path) if revs else None


def _is_chat_model(snap: Path) -> bool:
    """A generative model with weights and a chat template: what the agent
    loop can drive. Embedding and reranking models (the docs index's) fail."""
    try:
        config = json.loads((snap / "config.json").read_text())
    except (OSError, ValueError):
        return False
    if not any(a.endswith(_ARCH_SUFFIXES) for a in config.get("architectures") or []):
        return False
    if not any(snap.glob("*.safetensors")) and not any(snap.glob("*.bin")):
        return False
    if (snap / "chat_template.jinja").exists():
        return True
    try:
        return "chat_template" in json.loads((snap / "tokenizer_config.json").read_text())
    except (OSError, ValueError):
        return False


def local_chat_models(settings: Settings) -> list[dict]:
    """[{"id", "gb", "path"}] for every chat model in settings' HF cache."""
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir(settings.hf_cache_dir)
    except Exception:
        return []
    out = []
    for repo in info.repos:
        snap = _snapshot(repo) if repo.repo_type == "model" else None
        if snap and _is_chat_model(snap):
            out.append({"id": repo.repo_id, "gb": round(repo.size_on_disk / 1e9, 1), "path": str(snap)})
    return sorted(out, key=lambda m: m["id"])


def model_config(settings: Settings, model_id: str) -> dict:
    """config.json of a cached model ({} if not cached)."""
    for m in local_chat_models(settings):
        if m["id"] == model_id:
            try:
                return json.loads((Path(m["path"]) / "config.json").read_text())
            except (OSError, ValueError):
                return {}
    return {}


def free_gb(settings: Settings) -> float | None:
    path = settings.hf_cache_dir or Path.home() / ".cache" / "huggingface"
    while not path.exists() and path != path.parent:
        path = path.parent
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return None


def download(settings: Settings, model_id: str) -> str:
    """Download `model_id` into settings' HF cache; returns the snapshot path."""
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, cache_dir=settings.hf_cache_dir, token=settings.hf_token or None)
