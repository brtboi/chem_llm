"""Runtime configuration for chem_llm, loaded from a config.yaml.

Everything environment-specific -- API keys, where Quantum ESPRESSO lives,
where models and indexes are cached, where runs are written -- is resolved
here, so the rest of the package never reads an env var or assumes it is
running out of a source checkout.

Resolution order for each field, first hit wins:

    explicit argument  >  config.yaml  >  environment variable  >  default

Typical use::

    from chem_llm import configure
    configure("config.yaml")          # once, at process start

    from chem_llm import run_dft_agent
    run_dft_agent("TiO2", "P4_2/mnm")

`configure()` is optional: with no config file the settings fall back to
environment variables (ANTHROPIC_API_KEY, MP_API_KEY, HF_TOKEN, ...) and
paths under the current working directory, which is enough for a quick
start. See config.example.yaml for the full file format.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - yaml is a declared dependency
    yaml = None

try:  # a local .env is a convenient place for keys during development
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:  # pragma: no cover
    pass

# Filenames looked for, in order, when configure() is called with no path.
_CONFIG_FILENAMES = ("chem_llm.yaml", "config.yaml")


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


@dataclass
class QESettings:
    """How to invoke Quantum ESPRESSO on this machine.

    Either point `pw_x`/`bands_x` at absolute binaries, or give `module`
    (an Lmod module to load first) and leave the binaries as bare names to
    be found on PATH after the load. `mpi_prefix` is prepended verbatim,
    e.g. "srun -n 8" or "mpirun -np 4"; leave empty to run serially, which
    is what a single allocated node without an MPI launcher wants.
    """

    pw_x: str = "pw.x"
    bands_x: str = "bands.x"
    module: str | None = None
    mpi_prefix: str = ""
    # Pinned to 1 by default: a serial pw.x otherwise spawns one OpenMP
    # thread per core, which for a small cell is heavy oversubscription and
    # is far slower than running single-threaded.
    omp_num_threads: int = 1
    timeout_seconds: int = 7200

    def shell_command(self, exe: str, in_file: str, out_file: str) -> str:
        """Full shell command for one QE step, including any module load."""
        launch = f"{self.mpi_prefix} {exe}".strip()
        command = f"{launch} -in {in_file} > {out_file} 2>&1"
        if self.module:
            # Must be in the same shell invocation: an Lmod module is not
            # active in a fresh subprocess.
            command = f"module load {self.module} && {command}"
        return command


@dataclass
class ModelSettings:
    """Which LLM drives the agent loop."""

    backend: str = "claude"  # "claude" (API) or "qwen" (local transformers)

    # Claude API
    claude_model: str = "claude-opus-4-8"
    claude_max_tokens: int = 16000
    claude_effort: str = "high"

    # Local Qwen
    qwen_model: str = "Qwen/Qwen3.8-27B"
    max_new_tokens: int = 5000
    temperature: float = 0.0
    do_sample: bool = False


@dataclass
class AgentSettings:
    max_steps: int = 60
    max_history: int = 32
    read_max_chars: int = 10000


@dataclass
class Settings:
    # --- credentials ---
    anthropic_api_key: str | None = None
    mp_api_key: str | None = None
    hf_token: str | None = None

    # --- paths ---
    # Where runs are written: each run gets work_root/<name>/.
    work_root: Path = field(default_factory=lambda: Path.cwd() / "chem_llm_runs")
    # Hugging Face cache (sets HF_HOME before transformers is imported).
    hf_home: Path | None = None
    # Documentation retrieval index (built by chem_llm.retrieval).
    doc_index_dir: Path = field(default_factory=lambda: Path.cwd() / "retrieval_index")

    qe: QESettings = field(default_factory=QESettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)

    source: Path | None = None  # config.yaml this came from, if any

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any], source: Path | None = None) -> "Settings":
        data = dict(data or {})
        api = dict(data.get("api_keys") or {})
        paths = dict(data.get("paths") or {})

        def _path(value, default):
            return Path(value).expanduser().resolve() if value else default

        settings = cls(
            anthropic_api_key=api.get("anthropic") or _env("ANTHROPIC_API_KEY"),
            mp_api_key=api.get("materials_project") or _env("MP_API_KEY"),
            hf_token=api.get("huggingface") or _env("HF_TOKEN"),
            work_root=_path(paths.get("work_root"), Path.cwd() / "chem_llm_runs"),
            hf_home=_path(paths.get("hf_home") or _env("HF_HOME"), None),
            doc_index_dir=_path(paths.get("doc_index_dir") or _env("DOC_INDEX_DIR"), Path.cwd() / "retrieval_index"),
            qe=QESettings(**(data.get("quantum_espresso") or {})),
            model=ModelSettings(**(data.get("model") or {})),
            agent=AgentSettings(**(data.get("agent") or {})),
            source=source,
        )
        return settings

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        path = Path(path).expanduser().resolve()
        if yaml is None:
            raise RuntimeError("PyYAML is required to read a config file; pip install pyyaml")
        if not path.exists():
            raise FileNotFoundError(f"No chem_llm config at {path}")
        with path.open() as handle:
            return cls.from_dict(yaml.safe_load(handle) or {}, source=path)

    @classmethod
    def discover(cls, start: Path | None = None) -> "Settings":
        """Look for a config file in `start` (default cwd) and its parents;
        fall back to environment variables and defaults if none is found."""
        start = (start or Path.cwd()).resolve()
        for directory in (start, *start.parents):
            for name in _CONFIG_FILENAMES:
                candidate = directory / name
                if candidate.exists():
                    return cls.from_yaml(candidate)
        return cls.from_dict({})

    # ------------------------------------------------------------------
    def apply_env(self) -> None:
        """Export the few values libraries read straight from the
        environment. HF_HOME in particular must be set before transformers
        is imported, which is why loading settings early matters."""
        if self.hf_home:
            os.environ.setdefault("HF_HOME", str(self.hf_home))
        for key, value in (
            ("ANTHROPIC_API_KEY", self.anthropic_api_key),
            ("MP_API_KEY", self.mp_api_key),
            ("HF_TOKEN", self.hf_token),
        ):
            if value:
                os.environ.setdefault(key, value)

    def require(self, *fields: str) -> None:
        """Raise a single actionable error naming everything missing."""
        missing = [f for f in fields if not getattr(self, f, None)]
        if missing:
            where = f" (loaded from {self.source})" if self.source else " (no config file found)"
            raise RuntimeError(
                "Missing required chem_llm settings: "
                + ", ".join(missing)
                + where
                + ". Set them in config.yaml under api_keys, or as environment variables."
            )

    def with_overrides(self, **kwargs) -> "Settings":
        return replace(self, **kwargs)


# ----------------------------------------------------------------------
_SETTINGS: Settings | None = None


def configure(path: str | Path | None = None, **overrides) -> Settings:
    """Load settings for this process and make them the active ones."""
    settings = Settings.from_yaml(path) if path else Settings.discover()
    if overrides:
        settings = settings.with_overrides(**overrides)
    settings.apply_env()
    global _SETTINGS
    _SETTINGS = settings
    return settings


def get_settings() -> Settings:
    """Active settings, discovering a config file on first use."""
    global _SETTINGS
    if _SETTINGS is None:
        configure()
    return _SETTINGS


def set_settings(settings: Settings) -> None:
    global _SETTINGS
    _SETTINGS = settings
    settings.apply_env()
