"""Find, start and wait for a vLLM server for the agent's model.

Works on any machine with NVIDIA GPUs (a cloud VM on GCP, AWS or Azure, a
workstation, a SLURM node): it needs only `nvidia-smi` and a vLLM install.
vLLM pins its own torch, so it usually lives in its own venv; it is found as

    settings.model.vllm_executable  (config.yaml model.vllm_executable)
    $VLLM_VENV/bin/vllm, then `vllm` on PATH, then $PSCRATCH/venvs/vllm/bin/vllm (NERSC)

The server is started detached, so it outlives the chem-llm process and the
next run reuses it instead of paying the ~5 minute load again. It stops when
the machine or allocation does, or with `kill <pid>` (printed at start).

GPUs: with qe.gpu_ids set (dedicated QE GPUs), vLLM takes the other GPUs at
90% memory -- processes time-slicing a GPU run many times slower. Without
it, vLLM takes every GPU at 75% and leaves the rest to Quantum ESPRESSO.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

from .settings import Settings

MAX_MODEL_LEN = 131072  # agent prompts have reached ~71K tokens


class VLLMError(RuntimeError):
    pass


def find_vllm(settings: Settings) -> str | None:
    candidates = [settings.model.vllm_executable,
                  os.environ.get("VLLM_VENV") and str(Path(os.environ["VLLM_VENV"]) / "bin" / "vllm"),
                  shutil.which("vllm"),
                  os.environ.get("PSCRATCH") and str(Path(os.environ["PSCRATCH"]) / "venvs" / "vllm" / "bin" / "vllm")]
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    return None


def visible_gpus() -> list[int]:
    """GPU indices this process may use: CUDA_VISIBLE_DEVICES if set, else nvidia-smi's list."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [int(g) for g in env.split(",") if g.strip().isdigit()]
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [i for i, line in enumerate(l for l in out.splitlines() if l.startswith("GPU "))]


def plan_gpus(settings: Settings) -> tuple[list[int], float]:
    """(GPUs for vLLM, memory fraction). Tensor parallelism needs a power of
    two that divides the model's heads, so an odd leftover GPU stays idle."""
    gpus = visible_gpus()
    if not gpus:
        raise VLLMError("no NVIDIA GPU found (nvidia-smi); vLLM needs one -- use the huggingface backend "
                        "on a GPU machine, or Claude")
    qe = set(settings.qe.gpu_ids or [])
    mine = [g for g in gpus if g not in qe] if qe else gpus
    if not mine:
        raise VLLMError(f"every GPU ({gpus}) is reserved for Quantum ESPRESSO by qe.gpu_ids")
    n = 1 << (len(mine).bit_length() - 1)
    return mine[:n], 0.90 if qe else 0.75


def command(settings: Settings, model: str, model_config: dict | None = None) -> tuple[list[str], dict, list[int]]:
    """(argv, env, gpus) to serve `model` at settings.model.vllm_url."""
    exe = find_vllm(settings)
    if exe is None:
        raise VLLMError("vLLM not found. Install it (in its own venv: it pins its own torch), e.g.\n"
                        "    python -m venv ~/venvs/vllm && ~/venvs/vllm/bin/pip install vllm\n"
                        "then set model.vllm_executable in config.yaml (or VLLM_VENV) to it.")
    gpus, mem = plan_gpus(settings)
    cfg = model_config or {}
    text_cfg = cfg.get("text_config") or cfg
    max_len = min(MAX_MODEL_LEN, int(text_cfg.get("max_position_embeddings") or MAX_MODEL_LEN))
    port = urlparse(settings.model.vllm_url).port or 8000
    argv = [exe, "serve", model, "--port", str(port), "--tensor-parallel-size", str(len(gpus)),
            "--gpu-memory-utilization", str(mem), "--max-model-len", str(max_len), "--enable-prefix-caching"]
    if "vision_config" in cfg:
        # multimodal checkpoint, text-only agent: skip the vision encoder's memory reservation
        argv += ["--limit-mm-per-prompt", '{"image": 0, "video": 0}']

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)))
    if settings.hf_home:
        env["HF_HOME"] = str(settings.hf_home)
    if settings.hf_token:
        env["HF_TOKEN"] = settings.hf_token
    if os.environ.get("PSCRATCH"):
        # NERSC's home filesystem refuses the file locks vLLM/Triton caches take (OSError 524)
        cache = Path(os.environ["PSCRATCH"]) / ".cache"
        env.setdefault("XDG_CACHE_HOME", str(cache))
        env.setdefault("VLLM_CACHE_ROOT", str(cache / "vllm"))
        env.setdefault("TRITON_CACHE_DIR", str(cache / "triton"))
        env.setdefault("FLASHINFER_WORKSPACE_BASE", os.environ["PSCRATCH"])
    if model_config:
        env.setdefault("HF_HUB_OFFLINE", "1")  # already cached: do not phone home
    return argv, env, gpus


def served_at(url: str) -> list[str] | None:
    """Models the server at `url` serves, or None if nothing answers."""
    from .llm import check_server

    try:
        return check_server(url, timeout=3)
    except ConnectionError:
        return None


def served_models(settings: Settings) -> list[str] | None:
    return served_at(settings.model.vllm_url)


def start(settings: Settings, model: str, model_config: dict | None = None, log_dir: Path | None = None):
    """Start the server detached. Returns (Popen, log path, gpus)."""
    argv, env, gpus = command(settings, model, model_config)
    log_dir = Path(log_dir or settings.work_root / "_vllm")
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"vllm_{time.strftime('%Y%m%d_%H%M%S')}.log"
    with open(log, "w") as f:
        f.write("$ " + " ".join(argv) + f"\n# CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n")
        f.flush()
        proc = subprocess.Popen(argv, env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)  # survives chem-llm exiting
    (log_dir / "vllm.pid").write_text(str(proc.pid))
    return proc, log, gpus


def wait_until_ready(settings: Settings, model: str, proc, log: Path, timeout: float = 1800, on_tick=None):
    """Block until the server serves `model`; VLLMError (with the log's tail) if it dies or times out."""
    started = time.monotonic()
    while True:
        served = served_models(settings)
        if served is not None and model in served:
            return time.monotonic() - started
        if proc.poll() is not None:
            raise VLLMError(f"vLLM exited with code {proc.returncode} while loading; last lines of {log}:\n"
                            + tail(log))
        if time.monotonic() - started > timeout:
            raise VLLMError(f"vLLM did not come up within {timeout / 60:.0f} min; see {log}")
        if on_tick:
            on_tick(time.monotonic() - started, tail(log, 1))
        time.sleep(5)


def tail(path: Path, n: int = 15) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def ensure_server(settings: Settings, model: str, model_config: dict | None = None, console=None) -> None:
    """Reuse a server already serving `model` at vllm_url, else start one and wait."""
    served = served_models(settings)
    url = settings.model.vllm_url
    say = console.print if console else print
    if served is not None:
        if model in served:
            say(f"vLLM server at {url} is up, serving {model}.")
            return
        raise VLLMError(f"the vLLM server at {url} serves {served}, not {model}. Stop it, or pick that model, "
                        "or set model.vllm_url to another port.")
    proc, log, gpus = start(settings, model, model_config)
    say(f"Starting vLLM for {model} on GPU(s) {','.join(map(str, gpus))} (pid {proc.pid}); log: {log}")
    say(f"It keeps running after this run, for the next one to reuse. Stop it with: kill {proc.pid}")
    if console is not None:
        with console.status("loading the model (about 5 minutes) ...") as status:
            took = wait_until_ready(settings, model, proc, log, on_tick=lambda t, last: status.update(
                f"loading the model: {t / 60:.1f} min  [dim]{last[:90]}[/dim]"))
    else:
        took = wait_until_ready(settings, model, proc, log)
    say(f"vLLM ready after {took / 60:.1f} min.")
