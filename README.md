# chem_llm

[![PyPI](https://img.shields.io/pypi/v/chem-llm)](https://pypi.org/project/chem-llm/)
[![Python](https://img.shields.io/pypi/pyversions/chem-llm)](https://pypi.org/project/chem-llm/)
[![License: MIT](https://img.shields.io/pypi/l/chem-llm)](https://github.com/brtboi/chem_llm/blob/master/LICENSE)

An LLM agent that runs end-to-end DFT band-structure studies with Quantum ESPRESSO.

Give it a compound and a space group. It fetches the structure from the Materials Project,
reduces it to the primitive cell, builds a perturbed structure ensemble, writes the Quantum
ESPRESSO inputs, runs SCF + bands + post-processing, and plots the band structure. It
writes every script itself, and checks its own work against a set of invariants: electron
counts are read from `pw.out` rather than recalled, k-paths are derived from the actual cell,
units are converted rather than assumed.

```python
from chem_llm import DFTAgent

agent = DFTAgent()
result = agent.run("C", "Fd-3m", spacegroup_name="diamond")

print(result)        # C (Fd-3m): OK | 4.040 eV | 49 steps | 49.9 min | Qwen/Qwen3.8-27B
print(result.plot)   # chem_llm_runs/C/calculations/000/bands_000.pdf
```

## Install

Published on PyPI as [`chem-llm`](https://pypi.org/project/chem-llm/):

```bash
pip install chem-llm                 # the agent, local Qwen backend
pip install "chem-llm[claude]"       # + the Claude API backend
pip install "chem-llm[notebook]"     # + JupyterLab, to run main.ipynb
```

Python 3.12+. You also need:

- **Quantum ESPRESSO** — a working `pw.x` and `bands.x`. It is not a Python dependency.
- **A GPU** for the default local model (`Qwen/Qwen3.8-27B`, ~55 GB of bf16 weights; four
  40 GB A100s is comfortable). The Claude API backend needs no GPU but bills per step.
- **A Materials Project API key** in `MP_API_KEY` — the agent fetches every structure from there.

Keys are read from the environment, or from a `.env` file in the working directory:

```bash
MP_API_KEY=...
HF_TOKEN=...            # only needed for gated Hugging Face models
ANTHROPIC_API_KEY=...   # only for backend="claude"
```

## Configuration

An agent's configuration is a `Settings` object. Each `DFTAgent` owns a private copy, so
agents with different settings can coexist in one process — there is no global
configuration step.

### Defaults

`DFTAgent()` with no arguments runs with the setup this project was developed with on NERSC
Perlmutter. `DFTAgent.default_settings()` returns them as an editable object.

| | default |
|---|---|
| model | `Qwen/Qwen3.8-27B`, local (`backend="qwen"`) |
| generation | `max_new_tokens=5000`, greedy (`do_sample=False`) |
| agent loop | `max_steps=80`, `max_history=32`, `read_max_chars=10000` |
| Quantum ESPRESSO | `pw.x` / `bands.x`, serial, `OMP_NUM_THREADS=1`, 2 h timeout per step |
| QE module | `espresso/7.5-libxc-7.0.0-cpu` on Perlmutter, none elsewhere |
| model cache | `$HF_HOME`, else `$PSCRATCH/huggingface` on NERSC, else `~/.cache/huggingface` |
| docs index | `./retrieval_index/ase` |
| runs | `./chem_llm_runs/<compound>` |

The two machine-specific defaults turn on only when the environment identifies Perlmutter
(`NERSC_HOST`) or NERSC scratch (`$PSCRATCH`). On NERSC the model cache must not live on the
home filesystem: it rejects the file locks Hugging Face takes (`OSError: [Errno 524]`).

### Changing them

Three equivalent ways:

```python
from chem_llm import DFTAgent, Settings

# keyword overrides, for one-off changes
agent = DFTAgent(pw_x="/opt/qe/bin/pw.x", mpi_prefix="srun -n 8", max_steps=60)

# edit the defaults
settings = DFTAgent.default_settings()
settings.qe.module = None
settings.model.max_new_tokens = 4000
agent = DFTAgent(settings)

# a config file (https://github.com/brtboi/chem_llm/blob/master/config.example.yaml documents every field)
agent = DFTAgent(Settings.from_yaml("config.yaml"))
```

Precedence is **keyword argument > `Settings` field > environment / `.env` > default**.

`Settings` has three nested sections:

- `qe` — `QESettings`: `pw_x`, `bands_x`, `module`, `mpi_prefix`, `omp_num_threads`, `timeout_seconds`
- `model` — `ModelSettings`: `backend`, `qwen_model`, `max_new_tokens`, `temperature`, `do_sample`, and `claude_*`
- `agent` — `AgentSettings`: `max_steps`, `max_history`, `read_max_chars`

At the top level it holds the credentials (`mp_api_key`, `hf_token`, `anthropic_api_key`) and
the paths `hf_home`, `doc_index_dir` and `work_root`.

The agent copies the `Settings` it is given, so edit `agent.settings` once an agent exists.
Generation and loop parameters are read when each run starts, so they can change between runs
without reloading the model. The backend and model id are fixed once the model loads.

While a run executes, its agent's settings are what the tools see — how QE is invoked, the
Materials Project key, the docs index — and what the agent's own scripts inherit. That is
scoped to the run, not exported process-wide.

## Running

```python
agent = DFTAgent()
agent.load()                        # optional: pay the model-load cost up front

result = agent.run(
    "CsPbBr3", "Pm-3m",             # required: formula, space group (Materials Project notation)
    spacegroup_number=221,
    spacegroup_name="cubic perovskite",
    work_dir="runs/cspbbr3",        # default: settings.work_root / compound
    clear_dir=True,
    log_file="runs/cspbbr3.jsonl",  # default: work_dir / log.jsonl
    n_structures=51,                # size of the perturbed ensemble
    relativistic="scalar",          # or "fully" for spin-orbit
    functional="pbesol",
)
```

The model loads once and stays loaded, so a sweep pays the (multi-minute) load cost once:

```python
for compound, spacegroup in [("C", "Fd-3m"), ("MgO", "Fm-3m"), ("GaAs", "F-43m")]:
    print(agent.run(compound, spacegroup))
```

`run()` returns a `RunResult` read back off disk rather than from the agent's own claims:
`completed`, `scf_converged`, `dft_finished`, `n_atoms`, `n_electrons`, `band_gap` (eV,
computed from the bands output), `plot`, `summary` (the agent's own write-up), `num_steps`,
`runtime_seconds`, and the full agent `state`. `result.ok` is true only if the agent
finished **and** all three QE stages reported `JOB DONE` **and** a gap could be computed.

For a single run without keeping an agent around, `run_dft_agent("Si", "Fd-3m")` builds one,
runs it and discards it.

## Documentation index

The agent's `search_docs` tool searches a local index of the ASE and Quantum ESPRESSO reference
docs. Without it the agent works from memory of API signatures and namelist variables, which is
where most of its wasted steps come from. Every agent has one (`agent.docs`, at
`settings.doc_index_dir`); build it once per machine:

```python
docs = agent.docs
if not docs.available:
    docs.build()                              # scrapes + embeds; needs network, a few minutes

docs.search("ecutwfc units", top_k=3)         # the same retrieval the agent's tool runs
```

A different index: `DFTAgent(docs="path/to/index")` or `DocumentationIndex("path").build()`.

## Backends

`backend="qwen"` (the default) runs a local transformers model on your own GPU and costs
nothing per step. `backend="claude"` drives the same loop through the Anthropic API: same
prompt, same parsing, same tools, only the text generation differs. It needs the `claude`
extra (`pip install "chem-llm[claude]"`), `ANTHROPIC_API_KEY`, and **bills for every
step** — a full run is 40–60 API calls.

## Command line

```bash
chem-llm TiO2 P4_2/mnm --name rutile --work-dir runs/tio2
chem-llm C Fd-3m --pw-x /opt/qe/bin/pw.x --max-steps 60
chem-llm --build-docs                        # build the documentation index
chem-llm --help

chem-llm-replay sandbox/test27               # re-execute a logged run's tool calls, no LLM
chem-llm-replay-copy 17 25                   # replay sandbox/test17 into sandbox/test25
```

`chem-llm` reads `config.yaml` from the current directory or its parents when one exists
(or `--config PATH`), otherwise the defaults.

## Repository layout

The [source repository](https://github.com/brtboi/chem_llm) holds more than the package:

```
chem_llm/            the package
  agent.py           DFTAgent, DocumentationIndex         (public API)
  settings.py        Settings and its sections             (public API)
  pipeline.py        RunResult, run_dft_agent, output inspection
  tasks.py           the task prompt (build_task)
  agent_core.py      system prompt, agent loop, tool-call parsing
  claude_backend.py  Claude API backend
  tools/             the agent's tools: files, Python, QE, Materials Project, pseudopotentials,
                     docs search, QE input validation
  retrieval/         documentation scraping, chunking, BM25 + embedding search, reranking
  main.py, replay.py command-line entry points
  config.py          legacy module constants, used by the sandbox run scripts
main.ipynb           walkthrough: settings, docs index, one run, a sweep
config.example.yaml  every setting, documented
scripts/             build_doc_index.py, build_qe_namelists.py (regenerates tools/qe_namelists.py)
sandbox/             past experiment runs and their run_test*.py drivers
logs/                historical run logs
example/, deepseudopot_example/, deepseudopot_agent.ipynb, DEEPSEUDOPOT_HANDOFF.md
                     reference workflow + the paused DeePseudopot (neural pseudopotential) work
```

## Development

From a checkout of the repository:

```bash
uv sync                               # editable install + the dev group (Jupyter, anthropic)
uv build                              # -> dist/: wheel and sdist
python scripts/build_qe_namelists.py  # after a QE version bump: refresh the namelist table
```

[`main.ipynb`](https://github.com/brtboi/chem_llm/blob/master/main.ipynb) walks through settings, the documentation index, one run
and a sweep.

The (currently hidden) `train_deepseudopot` tool runs
[DeePseudopot](https://github.com/TommyLinkl/DeePseudopot), which has its own license and
is not shipped in the package. The repository vendors a checkout at
`chem_llm/DeePseudopot/`; with an installed package, set `DEEPSEUDOPOT_DIR` to a checkout.

## License

MIT — see [LICENSE](https://github.com/brtboi/chem_llm/blob/master/LICENSE).
