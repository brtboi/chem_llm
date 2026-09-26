"""chem_llm -- an LLM agent that runs end-to-end DFT band-structure studies.

Give it a compound and a space group; it fetches the structure from the
Materials Project, reduces it to the primitive cell, builds a perturbed
structure ensemble, writes Quantum ESPRESSO inputs, runs the SCF + bands +
post-processing chain, and plots the result -- writing every script itself
and checking its own work against a set of invariants.

    from chem_llm import configure, run_dft_agent

    configure("config.yaml")
    result = run_dft_agent("TiO2", "P4_2/mnm", spacegroup_name="rutile")
    print(result)

See config.example.yaml for the configuration file format.
"""
from pathlib import Path

# Kept for in-tree scripts and the sandbox runs, which anchor paths to the
# checkout. Anything installed from a wheel should use chem_llm.settings
# (work_root, doc_index_dir, ...) instead -- inside site-packages this
# points at the install directory, which is not a useful place to write.
REPO_ROOT = Path(__file__).resolve().parent.parent

from .settings import Settings, configure, get_settings, set_settings  # noqa: E402
from .tasks import build_task  # noqa: E402
from .pipeline import RunResult, run_dft_agent  # noqa: E402

__version__ = "0.1.0"

__all__ = [
    "REPO_ROOT",
    "RunResult",
    "Settings",
    "build_task",
    "configure",
    "get_settings",
    "run_dft_agent",
    "set_settings",
]
