"""The public interface: build an agent once, run it on many compounds.

    from chem_llm import DFTAgent, Settings

    agent = DFTAgent(Settings.from_yaml("config.yaml"))
    result = agent.run("TiO2", "P4_2/mnm")
    print(result)

Each agent owns its configuration: the Settings you pass in is copied, any
keyword overrides are applied to the copy, and that copy is what the model,
the tools and the QE runs see -- only while one of *this* agent's runs is
executing. Nothing is configured process-wide, so agents with different
settings can coexist in one process.

The model is loaded once, on the first run, and reused for every run after
it, so a sweep pays the (multi-minute) load cost once rather than per
compound.

The default backend is the local Qwen3 model, which runs on your own GPU
and costs nothing. The Claude API backend is opt-in (`backend="claude"`)
and spends real money per run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .pipeline import RunResult, _hide_tools, _inspect_outputs
from .settings import Settings, activated
from .tasks import build_task


class DocumentationIndex:
    """The corpus `search_docs` searches: ASE and Quantum ESPRESSO reference
    docs, indexed locally for hybrid keyword+semantic retrieval.

        docs = DocumentationIndex("./retrieval_index/ase")
        if not docs.available:
            docs.build()                 # scrapes; needs network, a few minutes

    An agent always has one -- built from its settings' doc_index_dir unless
    you pass DFTAgent(docs=...). Without a built index the agent still runs,
    but `search_docs` reports itself unavailable and the agent has to work
    from the model's memory of the APIs -- which is exactly when it tends to
    guess wrong function signatures.
    """

    def __init__(self, path: str | Path | None = None, sources=None, cache_dir: str | Path | None = None):
        """`path` defaults to the default settings' doc_index_dir
        (./retrieval_index/ase). `cache_dir` is the Hugging Face hub cache
        for the embedding and reranker models (None: the library default /
        $HF_HOME)."""
        if path is None:
            path = Settings().doc_index_dir
        self.path = Path(path).expanduser().resolve()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        # Imported lazily: the builder pulls in the scrapers (requests,
        # BeautifulSoup), which a run that only *queries* the index never needs.
        from .retrieval.build import DEFAULT_SOURCES

        self.sources = list(sources) if sources else list(DEFAULT_SOURCES)

    @classmethod
    def from_settings(cls, settings: Settings) -> "DocumentationIndex":
        return cls(settings.doc_index_dir, cache_dir=settings.hf_cache_dir)

    @property
    def available(self) -> bool:
        """True when a built index is present and loadable."""
        return (self.path / "chunks.jsonl").exists() and (self.path / "bm25.pkl").exists()

    @property
    def stats(self) -> dict:
        info = self.path / "build_info.json"
        if not info.exists():
            return {"available": self.available, "path": str(self.path)}
        import json

        data = json.loads(info.read_text())
        return {"available": self.available, "path": str(self.path), **data}

    def build(self, sources=None, embed: bool = True, **kwargs) -> dict:
        """Scrape and index the documentation sources into `self.path`
        (needs network). A few minutes; needed once per machine, and again
        whenever you want the corpus refreshed."""
        from .retrieval.build import build_index
        from .tools import docs as docs_tool

        info = build_index(sources or self.sources, index_dir=self.path, cache_dir=self.cache_dir, embed=embed, **kwargs)
        # Anything that already loaded the old index files should reload.
        docs_tool._retrievers.pop((self.path, self.cache_dir), None)
        return info

    def search(self, query: str, top_k: int = 5, sources=None) -> list[dict]:
        """Query the index directly -- the same retrieval the agent's
        `search_docs` tool runs."""
        from .tools.docs import get_retriever

        retriever = get_retriever(self.path, self.cache_dir)
        return [r.to_dict() for r in retriever.search(query, top_k=top_k, sources=sources)]

    def __repr__(self) -> str:
        return f"DocumentationIndex(path={str(self.path)!r}, available={self.available})"


@dataclass
class _LoadedModel:
    # (settings, verbose) -> step source. Taking settings per run means the
    # generation parameters are read when a run starts, so editing
    # agent.settings.model.max_new_tokens between runs takes effect without
    # reloading the weights.
    step_source_factory: Callable[[Settings, bool], Callable]
    name: str
    backend: str


class DFTAgent:
    """An agent that runs end-to-end DFT band-structure studies.

    `settings` describes the machine; `run()` describes the job.

    settings:   a Settings object, or the path to a config.yaml (see
                config.example.yaml). A Settings is copied, so editing your
                object afterwards does not change this agent -- edit
                `agent.settings` instead. Default:
                DFTAgent.default_settings() -- see there for the values.

    Keyword arguments override single fields of that copy, for the common
    cases where writing a Settings would be overkill:

    model:      HuggingFace id of the local model (settings.model.qwen_model).
    backend:    "qwen" (local, free) or "claude" (API, paid, opt-in).
    hf_home:    where model weights are cached. On a cluster point this at
                scratch -- the weights are tens of GB.
    hf_token, mp_api_key, anthropic_api_key:
                credentials; otherwise the Settings / environment values.
    pw_x, bands_x, qe_module, mpi_prefix, omp_num_threads, qe_timeout_seconds:
                how to run Quantum ESPRESSO (settings.qe).
    docs:       a DocumentationIndex or a path to one (settings.doc_index_dir).
    max_steps, max_history, max_new_tokens, temperature, do_sample:
                agent-loop and generation parameters.
    """

    def __init__(
        self,
        settings: Settings | str | Path | None = None,
        *,
        model: str | None = None,
        backend: str | None = None,
        hf_home: str | Path | None = None,
        hf_token: str | None = None,
        mp_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        pw_x: str | None = None,
        bands_x: str | None = None,
        qe_module: str | None = None,
        mpi_prefix: str | None = None,
        omp_num_threads: int | None = None,
        qe_timeout_seconds: int | None = None,
        docs: "DocumentationIndex | str | Path | None" = None,
        work_root: str | Path | None = None,
        max_steps: int | None = None,
        max_history: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        do_sample: bool | None = None,
        # Claude-only; ignored by the qwen backend.
        claude_model: str | None = None,
        claude_effort: str | None = None,
        claude_max_tokens: int | None = None,
    ):
        if settings is None:
            settings = self.default_settings()
        elif isinstance(settings, (str, Path)):
            settings = Settings.from_yaml(settings)
        self.settings = settings.copy()
        s = self.settings

        def override(section, **fields):
            for name, value in fields.items():
                if value is not None:
                    setattr(section, name, value)

        override(s, hf_token=hf_token, mp_api_key=mp_api_key, anthropic_api_key=anthropic_api_key)
        if hf_home is not None:
            s.hf_home = Path(hf_home).expanduser().resolve()
        if work_root is not None:
            s.work_root = Path(work_root).expanduser().resolve()
        override(s.qe, pw_x=pw_x, bands_x=bands_x, module=qe_module, mpi_prefix=mpi_prefix,
                 omp_num_threads=omp_num_threads, timeout_seconds=qe_timeout_seconds)
        override(s.model, backend=backend, qwen_model=model, max_new_tokens=max_new_tokens,
                 temperature=temperature, do_sample=do_sample, claude_model=claude_model,
                 claude_effort=claude_effort, claude_max_tokens=claude_max_tokens)
        override(s.agent, max_steps=max_steps, max_history=max_history)

        if isinstance(docs, DocumentationIndex):
            s.doc_index_dir = docs.path
        elif docs is not None:
            s.doc_index_dir = Path(docs).expanduser().resolve()

        self._loaded: _LoadedModel | None = None

    # ------------------------------------------------------------------
    @staticmethod
    def default_settings() -> Settings:
        """What DFTAgent() runs with when given no settings: the setup this
        project was developed with on NERSC Perlmutter.

            model           Qwen/Qwen3.8-27B, local ("qwen" backend)
            generation      max_new_tokens=5000, greedy (do_sample=False)
            agent loop      max_steps=80, max_history=32, read_max_chars=10000
            QE              pw.x / bands.x, serial, OMP_NUM_THREADS=1, 2 h timeout,
                            module espresso/7.5-libxc-7.0.0-cpu on Perlmutter
            model cache     $HF_HOME, else $PSCRATCH/huggingface on NERSC
            docs index      ./retrieval_index/ase
            runs            ./chem_llm_runs/<compound>
            credentials     MP_API_KEY / HF_TOKEN / ANTHROPIC_API_KEY from env or .env

        Returns a fresh object each call: edit it and pass it back in,
        e.g. `s = DFTAgent.default_settings(); s.qe.mpi_prefix = "srun -n 4";
        DFTAgent(s)`. No config file is read -- pass
        Settings.from_yaml(...) for that.
        """
        return Settings()

    @property
    def backend(self) -> str:
        return self._loaded.backend if self._loaded else self.settings.model.backend.lower()

    @property
    def model_name(self) -> str:
        if self._loaded:
            return self._loaded.name
        return self.settings.model.qwen_model if self.backend == "qwen" else self.settings.model.claude_model

    @property
    def docs(self) -> DocumentationIndex:
        """The documentation index this agent's search_docs tool searches."""
        return DocumentationIndex.from_settings(self.settings)

    # ------------------------------------------------------------------
    def load(self) -> None:
        """Download/load the model. Called automatically by the first
        `run()`; call it yourself to pay the cost up front.

        The backend and model id are fixed from here on: changing
        settings.model.backend / qwen_model afterwards needs a new agent.
        Generation parameters are still read at the start of each run.
        """
        if self._loaded is not None:
            return

        s = self.settings
        backend = s.model.backend.lower()

        if backend == "qwen":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from .agent_core import make_llm_step_source

            name = s.model.qwen_model
            # token=None downloads anonymously, which is all a public
            # checkpoint needs; a token only matters for gated repos.
            # cache_dir keeps this agent's weights under its own hf_home
            # without exporting HF_HOME for the whole process.
            hub = dict(token=s.hf_token or None, cache_dir=s.hf_cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(name, **hub)
            model = AutoModelForCausalLM.from_pretrained(name, device_map="auto", dtype=torch.bfloat16, **hub)

            def factory(settings: Settings, verbose: bool):
                return make_llm_step_source(
                    model, tokenizer, verbose,
                    max_new_tokens=settings.model.max_new_tokens,
                    temperature=settings.model.temperature,
                    do_sample=settings.model.do_sample,
                    max_history=settings.agent.max_history,
                )

            self._loaded = _LoadedModel(factory, name, backend)

        elif backend == "claude":
            s.require("anthropic_api_key")
            try:
                from .claude_backend import ClaudeModel, make_claude_step_source
            except ModuleNotFoundError as e:
                if e.name != "anthropic":
                    raise
                raise ModuleNotFoundError(
                    "The Claude backend needs the optional 'claude' extra: pip install 'chem-llm[claude]'"
                ) from e

            claude = ClaudeModel(
                model=s.model.claude_model,
                max_tokens=s.model.claude_max_tokens,
                effort=s.model.claude_effort,
                api_key=s.anthropic_api_key,
            )

            def factory(settings: Settings, verbose: bool):
                claude.max_tokens = settings.model.claude_max_tokens
                claude.effort = settings.model.claude_effort
                return make_claude_step_source(claude, verbose=verbose, max_history=settings.agent.max_history)

            self._loaded = _LoadedModel(factory, claude.model, backend)

        else:
            raise ValueError(f"Unknown backend {s.model.backend!r}; expected 'qwen' or 'claude'")

    # ------------------------------------------------------------------
    def run(
        self,
        compound: str,
        spacegroup: str,
        *,
        spacegroup_number: int | None = None,
        spacegroup_name: str = "",
        work_dir: str | Path | None = None,
        clear_dir: bool = True,
        log_file: str | Path | None = None,
        max_steps: int | None = None,
        n_structures: int = 51,
        relativistic: str = "scalar",
        functional: str = "pbesol",
        task: str | None = None,
        hide_tools=("train_deepseudopot",),
        verbose: bool = True,
    ) -> RunResult:
        """Run the full workflow for one compound.

        `compound` and `spacegroup` are the only required arguments;
        `spacegroup` uses Materials Project formatting ('P4_2/mnm',
        'Fd-3m'). Output goes to `work_dir`, defaulting to
        settings.work_root/<compound>.
        """
        # Removes the tools from the registry for the rest of the process --
        # agent_core serialises the tool list into the system prompt once, at
        # import time, so a later run cannot put them back.
        _hide_tools(hide_tools)
        self.load()

        from .agent_core import run_agent

        s = self.settings
        work_dir = Path(work_dir).expanduser().resolve() if work_dir else (s.work_root / compound).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

        prompt = task or build_task(
            compound=compound,
            spacegroup=spacegroup,
            spacegroup_number=spacegroup_number,
            spacegroup_name=spacegroup_name,
            n_structures=n_structures,
            relativistic=relativistic,
            functional=functional,
        )

        kwargs = dict(
            step_source=self._loaded.step_source_factory(s, verbose),
            model_name=self._loaded.name,
            work_dir=work_dir,
            max_steps=max_steps or s.agent.max_steps,
            verbose=verbose,
            clear_dir=clear_dir,
            load_path=None,
        )
        # run_agent's log_file has a sentinel default (work_dir/log.jsonl),
        # and log_file=None there means "do not log at all", so only pass it
        # through when the caller actually named a file.
        log_path = Path(log_file).expanduser().resolve() if log_file else None
        if log_path is not None:
            kwargs["log_file"] = log_path

        # run_agent chdir's into work_dir (every tool resolves paths against
        # cwd); restore it so a second run() from a notebook does not nest
        # run directories inside the first. The settings are active only for
        # this block: every tool call in the run sees this agent's config.
        previous_cwd = Path.cwd()
        try:
            with activated(s):
                state = run_agent(prompt, **kwargs)
        finally:
            os.chdir(previous_cwd)

        found = _inspect_outputs(work_dir)
        runtime = _runtime_from_log(log_path or work_dir / "log.jsonl")

        return RunResult(
            compound=compound,
            spacegroup=spacegroup,
            work_dir=work_dir,
            completed=state.done,
            num_steps=len(state.history),
            runtime_seconds=runtime,
            model=self._loaded.name,
            summary=state.final_result,
            scf_converged=found.get("scf_converged", False),
            dft_finished=found.get("dft_finished", False),
            n_atoms=found.get("n_atoms"),
            n_electrons=found.get("n_electrons"),
            band_gap=found.get("band_gap"),
            plot=found.get("plot"),
            state=state,
        )

    def __repr__(self) -> str:
        loaded = "loaded" if self._loaded else "not loaded"
        return f"DFTAgent(backend={self.backend!r}, model={self.model_name!r}, {loaded})"


def _runtime_from_log(log_file: Path) -> float:
    if not log_file.exists():
        return 0.0
    import json

    try:
        return json.loads(log_file.read_text().strip().splitlines()[-1]).get("runtime", 0.0)
    except (ValueError, IndexError):
        return 0.0
