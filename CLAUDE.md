# chem_llm

An LLM agent (local Qwen3 by default) that writes and runs its own Python scripts to
fetch a structure from Materials Project, build a perturbed ensemble, run Quantum
ESPRESSO, and plot a band structure. See README.md for the package API.

## Where things live
- Public API: `DFTAgent` / `DocumentationIndex` in `chem_llm/agent.py`, `Settings` in
  `chem_llm/settings.py`. Each agent owns its Settings; tools read the running agent's
  settings via `settings.active_settings()` — there is no global configuration.
- The agent's behavior is driven by `SYSTEM_PROMPT_TEMPLATE` in `chem_llm/agent_core.py`
  (AGENT PROTOCOL / INVARIANTS / DECISION RULES) plus the task text in `chem_llm/tasks.py`.
  Most physics fixes are prompt fixes: an invariant that says "derive X from the actual
  structure/output in code, never from memory".
- `chem_llm/config.py` is legacy module constants, still used by the `sandbox/run_test*.py`
  drivers and as defaults for `agent_core` functions called directly.

## Backends and run records
- `model.backend: vllm` runs the same Qwen weights on a vLLM server (`scripts/serve_vllm.sh`,
  own venv at `$PSCRATCH/venvs/vllm`): ~5 s/step vs ~50 s in-process; rutile TiO2 end to end
  in ~6 min. The server must be started on the node the runs use.
- The loop is a LangGraph graph (`chem_llm/graph.py`); backends are LangChain chat models
  (`chem_llm/llm.py`). Each run logs `metrics` (time per phase), `prompt` (version hash +
  system prompt) and `outcome`; `trace.jsonl` is written per step. `chem-llm --diff-runs A B`.
- `pytest` (no GPU/network; `chem_llm.testing` scripts the model and mocks QE) after any change.

## Checking a run's physics
A band plot can look smooth and plausible while being wrong. Check degeneracies at the
labelled high-symmetry points (e.g. diamond at X: two doubly-degenerate valence pairs) and
compare the gap against a known PBE value, rather than trusting the plot or the agent's summary.

## Constraints
- Default every run to the local Qwen backend. Never use the Claude API backend (it bills
  the user's credits) without explicit instruction.
- Do not delete anything under `sandbox/` or `logs/`.

## NERSC Perlmutter infrastructure
Runs happen inside interactive SLURM GPU allocations that expire and kill background jobs;
check `squeue --me` for time left before starting a long run (~45–60 min per compound).

Anything that takes file locks under `$HOME` (uv, HF, vLLM/Triton/flashinfer caches) fails
with OSError 524: point its cache at `$PSCRATCH`.

Stalls of 30+ min at trivial steps (e.g. `import numpy`) with `/proc/<pid>/wchan` =
`cl_sync_io_wait` and a node load average of 60–160 are shared-node Lustre contention, not a
code bug. Confirm the session is on an exclusive node before assuming a fix is broken — a
Claude Code terminal stays on the node it started on.
