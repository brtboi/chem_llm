"""chem_llm -- an LLM agent that runs end-to-end DFT band-structure studies.

Give it a compound and a space group; it fetches the structure from the
Materials Project, reduces it to the primitive cell, builds a perturbed
structure ensemble, writes Quantum ESPRESSO inputs, runs the SCF + bands +
post-processing chain, and plots the result -- writing every script itself
and checking its own work against a set of invariants.

    from chem_llm import DFTAgent, Settings

    agent = DFTAgent(Settings.from_yaml("config.yaml"))    # the machine
    result = agent.run("TiO2", "P4_2/mnm")                 # the job
    print(result)

Each DFTAgent owns its Settings (credentials, Quantum ESPRESSO, caches,
generation parameters); nothing is configured process-wide. The agent holds
the model, so a second `run()` reuses the loaded weights.

See config.example.yaml for the configuration file format.
"""
import importlib
import importlib.abc
import importlib.util
import sys
from pathlib import Path

# Kept for in-tree scripts and the sandbox runs, which anchor paths to the
# checkout. Anything installed from a wheel should use chem_llm.settings
# (work_root, doc_index_dir, ...) instead -- inside site-packages this
# points at the install directory, which is not a useful place to write.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Where modules lived before the package was split into subpackages
# (2026-11-12). Scripts in archived runs (bench/runs, sandbox) import the old
# paths -- `from chem_llm.kpath import plot_axis` -- and chem-llm-replay
# re-executes them, so the old names still import, as the moved module itself.
MOVED_MODULES = {
    "agent_core": "core.agent_core", "graph": "core.graph", "state": "core.state",
    "history": "core.history", "tasks": "core.tasks", "pipeline": "core.pipeline", "config": "core.config",
    "llm": "llm.models", "claude_backend": "llm.claude_backend", "vllm_server": "llm.vllm_server",
    "local_models": "llm.local_models",
    "kpath": "dft.kpath", "kpath_check": "dft.kpath_check", "qe_output": "dft.qe_output",
    "polymorphs": "dft.polymorphs",
    "dpp_agent": "dpp.agent", "dpp_io": "dpp.io",
    "runlog": "records.runlog", "telemetry": "records.telemetry", "archive": "records.archive",
    "report": "records.report", "rundiff": "records.rundiff", "replay": "records.replay",
    "main": "cli.main", "interactive": "cli.interactive", "progress": "cli.progress",
}


class _MovedModule(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """`import chem_llm.kpath` -> the chem_llm.dft.kpath module object."""

    def find_spec(self, name, path=None, target=None):
        old = name[len(__name__) + 1:] if name.startswith(__name__ + ".") else None
        # chem_llm.llm is now a package; only its old *module* name is aliased elsewhere
        if old in MOVED_MODULES and old != "llm":
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        module = importlib.import_module(f"{__name__}.{MOVED_MODULES[spec.name.rsplit('.', 1)[1]]}")
        self._own_spec = module.__spec__
        return module

    def exec_module(self, module):
        # the import system set __spec__ to the alias's spec; put the module's own back
        module.__spec__ = self._own_spec


sys.meta_path.append(_MovedModule())

from .settings import AgentSettings, ModelSettings, QESettings, Settings  # noqa: E402
from .core.tasks import build_task  # noqa: E402
from .core.pipeline import RunResult, run_dft_agent  # noqa: E402
from .agent import DFTAgent, DocumentationIndex  # noqa: E402

__version__ = "0.1.0"

__all__ = [
    "AgentSettings",
    "DFTAgent",
    "DocumentationIndex",
    "ModelSettings",
    "QESettings",
    "REPO_ROOT",
    "RunResult",
    "Settings",
    "build_task",
    "run_dft_agent",
]
