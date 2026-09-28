"""Configuration for a chem_llm agent.

A `Settings` object holds everything machine-specific -- credentials, how
to invoke Quantum ESPRESSO, where model weights and the documentation index
are cached, where runs are written, and how the model generates. Each
DFTAgent owns its own copy. There is no process-wide configuration step:

    from chem_llm import DFTAgent, Settings

    settings = Settings.from_yaml("config.yaml")
    settings.qe.mpi_prefix = "srun -n 4"

    agent = DFTAgent(settings)

Resolution order for each field, first hit wins:

    DFTAgent keyword argument  >  Settings field  >  environment / .env  >  default

The defaults are the configuration this project was developed with on NERSC
Perlmutter, so `DFTAgent()` with no arguments reproduces it:

    model           Qwen/Qwen3.8-27B (local; backend "qwen")
    generation      max_new_tokens=5000, greedy (do_sample=False)
    agent loop      max_steps=80, max_history=32, read_max_chars=10000
    Quantum ESPRESSO pw.x / bands.x, serial, OMP_NUM_THREADS=1, 2 h timeout,
                    module espresso/7.5-libxc-7.0.0-cpu  (on Perlmutter only)
    model cache     $HF_HOME, else $PSCRATCH/huggingface (on NERSC), else ~/.cache
    docs index      ./retrieval_index/ase
    runs            ./chem_llm_runs/<compound>

The two machine-specific ones (QE module, scratch cache) switch on only when
the environment says you are on Perlmutter / NERSC, so elsewhere the defaults
are the plain, portable ones. Credentials always come from the environment
(including a local .env) unless given explicitly, so `Settings()` alone is
enough when your keys live in .env. See config.example.yaml for the file
format.

While an agent is running, its settings are the *active* settings: the tools
it calls (run_espresso_workflow, generate_cif, read_file, search_docs) read
them through `active_settings()`. That is scoped to the run -- set on entry,
restored on exit -- so two agents in one process never see each other's
configuration.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - yaml is a declared dependency
    yaml = None

try:  # a local .env is a convenient place for keys during development
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:  # pragma: no cover
    pass

# Filenames looked for, in order, by Settings.discover().
_CONFIG_FILENAMES = ("chem_llm.yaml", "config.yaml")

_CREDENTIAL_ENV = {
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "mp_api_key": "MP_API_KEY",
    "hf_token": "HF_TOKEN",
}


def _env(name: str) -> str | None:
    return os.environ.get(name) or None


def _path(value) -> Path | None:
    return Path(value).expanduser().resolve() if value else None


# The Quantum ESPRESSO build this project was developed against.
PERLMUTTER_QE_MODULE = "espresso/7.5-libxc-7.0.0-cpu"


def _default_qe_module() -> str | None:
    return PERLMUTTER_QE_MODULE if os.environ.get("NERSC_HOST") == "perlmutter" else None


def _default_hf_home() -> Path | None:
    # On NERSC the home filesystem is small and refuses the file locks
    # huggingface_hub takes (OSError 524), so default the cache onto scratch.
    scratch = _env("PSCRATCH")
    return Path(scratch) / "huggingface" if scratch else None


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
    # Lmod module loaded before each QE step. Default: the Perlmutter build
    # this project used, when running on Perlmutter; otherwise none.
    module: str | None = field(default_factory=_default_qe_module)
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
    """Which LLM drives the agent loop, and how it generates."""

    # Local Qwen by default: it runs on the machine you already have and
    # costs nothing per run. "claude" hits the Anthropic API and bills.
    backend: str = "qwen"  # "qwen" (local transformers) or "claude" (API)

    # Local Qwen
    qwen_model: str = "Qwen/Qwen3.8-27B"
    max_new_tokens: int = 5000
    temperature: float = 0.0   # only used when do_sample is true
    do_sample: bool = False

    # Claude API
    claude_model: str = "claude-opus-4-8"
    claude_max_tokens: int = 16000
    claude_effort: str = "high"


@dataclass
class AgentSettings:
    max_steps: int = 80
    # Tool results kept verbatim in the prompt; older ones are summarised.
    max_history: int = 32
    # read_file truncates to this many characters.
    read_max_chars: int = 10000


@dataclass
class Settings:
    # --- credentials (None -> environment / .env) ---
    anthropic_api_key: str | None = None
    mp_api_key: str | None = None
    hf_token: str | None = None

    # --- paths ---
    # Where runs are written by default: work_root/<compound>/.
    work_root: Path = field(default_factory=lambda: Path.cwd() / "chem_llm_runs")
    # Hugging Face cache root (the directory HF_HOME would name). None ->
    # $HF_HOME, else $PSCRATCH/huggingface on NERSC, else the library
    # default (~/.cache/huggingface).
    hf_home: Path | None = None
    # Documentation retrieval index (DocumentationIndex.build()).
    doc_index_dir: Path = field(default_factory=lambda: Path.cwd() / "retrieval_index" / "ase")

    qe: QESettings = field(default_factory=QESettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)

    source: Path | None = None  # config.yaml this came from, if any

    def __post_init__(self):
        for attr, env_name in _CREDENTIAL_ENV.items():
            if not getattr(self, attr):
                setattr(self, attr, _env(env_name))
        self.hf_home = _path(self.hf_home or _env("HF_HOME") or _default_hf_home())
        self.work_root = _path(self.work_root)
        self.doc_index_dir = _path(self.doc_index_dir)

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any], source: Path | None = None) -> "Settings":
        """Build from a parsed config dict. Relative paths are taken relative
        to the config file (`source`) when there is one, so a config means
        the same thing whichever directory it is loaded from."""
        data = dict(data or {})
        api = dict(data.get("api_keys") or {})
        paths = dict(data.get("paths") or {})
        if source is not None:
            base = Path(source).parent
            for key, value in paths.items():
                if value and not Path(value).expanduser().is_absolute():
                    paths[key] = base / value

        kwargs: dict[str, Any] = dict(
            anthropic_api_key=api.get("anthropic"),
            mp_api_key=api.get("materials_project"),
            hf_token=api.get("huggingface"),
            hf_home=paths.get("hf_home"),
            qe=QESettings(**(data.get("quantum_espresso") or {})),
            model=ModelSettings(**(data.get("model") or {})),
            agent=AgentSettings(**(data.get("agent") or {})),
            source=source,
        )
        # Left out rather than passed as None so the dataclass defaults
        # (relative to cwd) apply when the file does not name them.
        if paths.get("work_root"):
            kwargs["work_root"] = paths["work_root"]
        if paths.get("doc_index_dir") or _env("DOC_INDEX_DIR"):
            kwargs["doc_index_dir"] = paths.get("doc_index_dir") or _env("DOC_INDEX_DIR")
        return cls(**kwargs)

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
        """Load the first chem_llm.yaml / config.yaml found in `start`
        (default cwd) or its parents; with none, environment and defaults."""
        start = (start or Path.cwd()).resolve()
        for directory in (start, *start.parents):
            for name in _CONFIG_FILENAMES:
                candidate = directory / name
                if candidate.exists():
                    return cls.from_yaml(candidate)
        return cls()

    # ------------------------------------------------------------------
    def copy(self) -> "Settings":
        """An independent copy, nested sections included."""
        return deepcopy(self)

    def with_overrides(self, **kwargs) -> "Settings":
        """A copy with top-level fields replaced. The nested qe/model/agent
        sections are deep-copied too, so editing the result never reaches
        back into this object."""
        return replace(self.copy(), **kwargs)

    @property
    def hf_cache_dir(self) -> Path | None:
        """The hub cache inside hf_home -- what transformers' `cache_dir`
        and sentence-transformers' `cache_folder` expect."""
        return self.hf_home / "hub" if self.hf_home else None

    def subprocess_env(self) -> dict[str, str]:
        """Environment for processes a tool launches on this agent's behalf
        (agent-written scripts, pw.x). These settings win over the parent
        process's environment, so a key passed to one DFTAgent reaches that
        agent's scripts without being exported process-wide."""
        env = dict(os.environ)
        for attr, env_name in _CREDENTIAL_ENV.items():
            value = getattr(self, attr)
            if value:
                env[env_name] = value
        if self.hf_home:
            env["HF_HOME"] = str(self.hf_home)
        return env

    def require(self, *fields: str) -> None:
        """Raise a single actionable error naming everything missing."""
        missing = [f for f in fields if not getattr(self, f, None)]
        if missing:
            where = f" (loaded from {self.source})" if self.source else ""
            env = ", ".join(_CREDENTIAL_ENV.get(f, f.upper()) for f in missing)
            raise RuntimeError(
                "Missing required chem_llm settings: " + ", ".join(missing) + where
                + ". Pass them to Settings/DFTAgent, set them in config.yaml under api_keys, "
                f"or export {env} (a .env file works)."
            )

    def __repr__(self) -> str:
        # Never print credentials, only whether they are present.
        keys = {attr: bool(getattr(self, attr)) for attr in _CREDENTIAL_ENV}
        return (
            f"Settings(source={str(self.source) if self.source else None!r}, keys={keys}, "
            f"work_root={str(self.work_root)!r}, hf_home={str(self.hf_home) if self.hf_home else None!r}, "
            f"doc_index_dir={str(self.doc_index_dir)!r}, qe={self.qe}, model={self.model}, agent={self.agent})"
        )


# ----------------------------------------------------------------------
# Active settings: those of the agent run currently executing.

_ACTIVE: ContextVar[Settings | None] = ContextVar("chem_llm_active_settings", default=None)
_FALLBACK: Settings | None = None


@contextmanager
def activated(settings: Settings) -> Iterator[Settings]:
    """Make `settings` the active ones for the duration of the block, then
    restore whatever was active before. DFTAgent.run() wraps each run in
    this; you only need it to call tools directly with a given Settings."""
    token = _ACTIVE.set(settings)
    try:
        yield settings
    finally:
        _ACTIVE.reset(token)


def active_settings() -> Settings:
    """The settings tools should use right now.

    Inside DFTAgent.run(): that agent's settings. Outside any run -- a tool
    called by hand, or the sandbox scripts that drive agent_core.run_agent
    directly -- whatever Settings.discover() finds, computed once. That
    fallback is read-only: nothing in the package sets it.
    """
    settings = _ACTIVE.get()
    if settings is not None:
        return settings
    global _FALLBACK
    if _FALLBACK is None:
        _FALLBACK = Settings.discover()
    return _FALLBACK
