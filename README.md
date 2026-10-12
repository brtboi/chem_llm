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
| generation | `max_new_tokens=5000`, sampled at `temperature=0.3` (`do_sample=True`); set `do_sample=False` for reproducible greedy runs |
| agent loop | `max_steps=80`, `max_history=32`, `trim_history=True`, `read_max_chars=10000` |
| Quantum ESPRESSO | `pw.x` / `bands.x`, `OMP_NUM_THREADS=1`, 2 h timeout per step |
| QE on Perlmutter | GPU build `espresso/7.5-libxc-7.0.0-gpu-cu13`, one MPI rank per GPU: `srun -n 4 -c 32 --gpus-per-task=1 --gpu-bind=closest pw.x -nk 4` |
| QE elsewhere | serial, no module |
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
- `agent` — `AgentSettings`: `max_steps`, `max_history`, `trim_history`, `read_max_chars`

At the top level it holds the credentials (`mp_api_key`, `hf_token`, `anthropic_api_key`) and
the paths `hf_home`, `doc_index_dir` and `work_root`.

**History trimming.** The prompt shows the agent its last `max_history` tool calls. With
`trim_history=True` (the default) it drops the provably dead parts of those calls:

- note entries, which are already in the Notes section;
- versions of a file that were later rewritten, and reads that were later repeated;
- once a script runs successfully, its failed attempts and the doc searches made to fix them.
  These collapse to a single line that keeps the error that was fixed.

It also adds a Progress ledger of which scripts ran, which are failing, the QE status and the
files read. Nothing is hidden just for being old: an output the agent may still be using stays
until it's superseded or leaves the window. Replayed over eight logged runs, it cut the
median prompt 16% and the worst 19% (71K to 58K tokens). Only the prompt changes; the logged
history stays complete.

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

Every backend is a LangChain chat model, and the agent loop is a LangGraph graph
(`chem_llm/graph.py`: `think` asks the model for one tool call, `act` runs it). Only the
text generation differs between backends:

| backend | what generates | cost |
|---|---|---|
| `qwen` (default) | the local model, loaded into this process with transformers | free; ~50 s per step for the 27B model |
| `vllm` | the same local model on a [vLLM](https://docs.vllm.ai) server | free; several times faster per step |
| `claude` | the Anthropic API (`pip install "chem-llm[claude]"`, `ANTHROPIC_API_KEY`) | **bills every step**; a run is 40–60 calls |

Any other LangChain chat model can drive the agent too: `DFTAgent(settings, llm=model)`.

### vLLM

vLLM batches requests, uses paged attention and caches the prompt prefix. Every step
re-sends the same ~8K-token system prompt, and prefix caching computes it once. One
server can serve several runs at once. vLLM pins its own torch, so it gets its own venv:

```bash
uv venv --python 3.12 --managed-python $PSCRATCH/venvs/vllm
VIRTUAL_ENV=$PSCRATCH/venvs/vllm uv pip install vllm
scripts/serve_vllm.sh > vllm.log 2>&1 &     # ~5 min to load; serves on localhost:8000
chem-llm TiO2 P4_2/mnm --backend vllm
```

(On NERSC, put uv's state on scratch first, `export UV_CACHE_DIR=$PSCRATCH/.uv-cache
UV_PYTHON_INSTALL_DIR=$PSCRATCH/.uv-python`: the home filesystem refuses file locks. The
serve script points vLLM's caches at scratch for the same reason.) The server takes 75% of
each GPU's memory by default, which leaves room for Quantum ESPRESSO on the same GPUs. Set
`GPU_MEM` to change that, and `MODEL`, `PORT` and `TP` for the other knobs. Set
`model.vllm_url` in `config.yaml` (or `--vllm-url`) if the server is not on
`localhost:8000`.

### Tracing

LangChain can send every run (the graph, each step, each model call with its prompt) to
[LangSmith](https://smith.langchain.com): `export LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...`. It is off unless both are set. Without it, each run already logs its
own timings and prompt version (see [Run records](#run-records)).

## Command line

```bash
chem-llm TiO2 P4_2/mnm --name rutile --work-dir runs/tio2
chem-llm TiO2 P4_2/mnm --pretty-progress                # readable live stream
chem-llm TiO2 P4_2/mnm --summarize > summary.txt        # stdout: just the summary at the end
chem-llm TiO2 P4_2/mnm --pretty-progress --summarize    # both: progress on stderr, summary on stdout
chem-llm C Fd-3m --pw-x /opt/qe/bin/pw.x --max-steps 60
chem-llm TiO2 P4_2/mnm --backend vllm                  # model on a vLLM server (much faster)
chem-llm --summarize-log runs/tio2                      # summary of a past run (no GPU needed)
chem-llm --diff-runs runs/tio2@0 runs/tio2              # compare two recorded runs
chem-llm --build-docs                                   # build the documentation index
chem-llm --help

chem-llm-replay sandbox/test27               # re-execute a logged run's tool calls, no LLM
chem-llm-replay-copy 17 25                   # replay sandbox/test17 into sandbox/test25
```

`chem-llm` reads `config.yaml` from the current directory or its parents when one exists
(or `--config PATH`), otherwise the defaults.

### Flags

| flag | what it does |
|---|---|
| `compound spacegroup` | required: e.g. `TiO2 P4_2/mnm`, `C P6_3/mmc` (Materials Project notation) |
| `--spacegroup-number N`, `--name NAME` | optional: help pick the structure / label it in the prompt |
| `--work-dir DIR` | where to run (default `<work_root>/<compound>`) |
| `--keep` | keep existing files; **by default the work dir is cleared, except `*.jsonl` logs** |
| `--log-file F` | run record somewhere other than `<work_dir>/log.jsonl` |
| `--pretty-progress` | readable live stream instead of raw model output (below) |
| `--summarize` | print the run summary when the run ends; the live stream goes to stderr |
| `--quiet` | print nothing during the run |
| `--model`, `--backend {qwen,vllm,claude}` | the model; `claude` bills per step |
| `--vllm-url URL` | the vLLM server (default `http://localhost:8000/v1`) |
| `--max-steps N`, `--structures N` | step limit (80), size of the perturbed ensemble (51) |
| `--pw-x`, `--bands-x`, `--qe-module` | where Quantum ESPRESSO comes from |
| `--summarize-log PATH [--index N]` | summary of a recorded run (work dir or `log.jsonl`), then exit |
| `--diff-runs RUN_A RUN_B` | compare two recorded runs (`dir`, `dir@N` for the N-th record), then exit |
| `--build-docs` | build the documentation search index, then exit |

Decoding (`temperature`, `do_sample`), history (`max_history`, `trim_history`) and the QE
launch (`mpi_prefix`, `pw_args`) have no flags: set them in `config.yaml`.

### Live output

By default the CLI streams every step (the model's raw output, the tool call, the full tool
result). `--pretty-progress` streams one readable line per action instead, with the agent's
notes:

```
[16 | 09:30] fetching a norm-conserving, scalar-relativistic PBEsol pseudopotential (stringent) for Ti ...
             -> z_valence 12, recommended ecutwfc 42 Ha
[21 | 13:02] running setup_jobs.py ...
             -> FAILED: NameError: name 'Ecutwfc_RY' is not defined
```

It is formatted from the tool calls and their results, so the model does nothing differently.
In Python: `agent.run(..., verbose=False, on_step=chem_llm.progress.ProgressPrinter())`.

### Run summary

`--summarize` prints a summary when the run finishes; everything streamed during the run goes
to stderr, so `> file` captures only the summary. `--summarize-log PATH` prints the same
summary for any recorded run (`--index N` picks an earlier run in that log). It has five
sections:

1. **What the agent did:** step by step, the Materials Project and Pseudo-Dojo data it
   pulled, every script it wrote (how many versions and runs, and how each ended), each
   validation, and its Quantum ESPRESSO runs.
2. **DFT setup**, written for a physicist or chemist reproducing the calculation:
   - structure: source, the cell used and its positions;
   - pseudopotentials: type, relativistic treatment, functional, accuracy table, valence,
     recommended cutoff;
   - pw.x settings: exchange-correlation, spin-orbit, cutoffs, occupations, k-mesh,
     convergence threshold;
   - bands: the k-path (named by symmetry), nbnd, the electron count, the band gap;
   - the perturbed-structure ensemble (displacements measured from the structures);
   - the Quantum ESPRESSO version and parallel setup.
3. **Agent and harness**, for CS / LLM engineers: the base model and decoding settings, the
   run's tool-call counts, how the prompt is built, the tools, context management, the
   harness guards, and the QE launch command. Runs record their configuration for this
   section.
4. **Errors, uncertainties and difficulties:**
   - failed runs and the errors fixed, validator findings, rewrite loops, and QE inputs
     regenerated without re-running QE;
   - where the documentation lookups clustered, and calls the harness refused;
   - incomplete k-path labels;
   - notes where the agent flagged doubt, and its own stated assumptions and review items.
5. **Where the time went:** wall time split into model generation, Quantum ESPRESSO, the
   agent's scripts and the other tools; model calls and tokens; QE time per stage (SCF,
   bands, bands.x); time per workflow STEP, taken from the agent's checkpoint notes.

It is built deterministically from the run's log and the files in its work dir, so every value
is read from a tool result or a file, never generated. In Python, the same summary is
`result.report()`.

## Run records

Each run appends one JSON record to `<work_dir>/log.jsonl`. Alongside the full history, the
record holds:

| field | contents |
|---|---|
| `run_id` | unique id, also on every line of the trace |
| `config` | backend, model, decoding, history settings, QE launch command |
| `prompt` | the full system prompt, and `version`, a hash of the system prompt and task |
| `metrics` | seconds per phase (`llm`, `quantum_espresso`, `agent_scripts`, ...), per tool, per QE stage and per workflow STEP; prompt/completion tokens; tokens/s |
| `outcome` | band gap, SCF convergence, electron count and plot, read back from disk |
| `final_state.steps` | one record per step: model seconds, tool seconds, prompt and completion tokens |

`<work_dir>/trace.jsonl` gets the same per-step records as each step finishes, so a run
killed mid-way (for example by an expired allocation) still leaves its trajectory.

`chem-llm --diff-runs A B` puts two runs side by side: outcome, steps, time per phase,
tokens, tool calls, the settings that differ, and a unified diff of the system prompt and
task. Runs with the same prompt `version` saw identical instructions.

## Repository layout

The [source repository](https://github.com/brtboi/chem_llm) holds more than the package:

```
chem_llm/            the package
  agent.py           DFTAgent, DocumentationIndex         (public API)
  settings.py        Settings and its sections             (public API)
  pipeline.py        RunResult, run_dft_agent, output inspection
  tasks.py           the task prompt (build_task)
  agent_core.py      system prompt, step sources, tool execution and refusals, run logging
  graph.py           the agent loop as a LangGraph graph
  llm.py             the backends as LangChain chat models (transformers, vLLM, Claude)
  claude_backend.py  Claude API client and its error policy
  telemetry.py       per-step timing, trace.jsonl, run metrics
  rundiff.py         prompt versions, --diff-runs
  testing.py         scripted model and tool mocks for running the loop without a GPU
  tools/             the agent's tools: files, Python, QE, Materials Project, pseudopotentials,
                     docs search, QE input validation
  retrieval/         documentation scraping, chunking, BM25 + embedding search, reranking
  main.py, replay.py command-line entry points
  config.py          legacy module constants, used by the sandbox run scripts
main.ipynb           walkthrough: settings, docs index, one run, a sweep
config.example.yaml  every setting, documented
tests/               pytest suite (scripted model, QE mocked; a live vLLM test is skipped without a server)
scripts/             serve_vllm.sh, build_doc_index.py, build_qe_namelists.py (regenerates tools/qe_namelists.py)
sandbox/             past experiment runs and their run_test*.py drivers
logs/                historical run logs
example/, deepseudopot_example/, deepseudopot_agent.ipynb, DEEPSEUDOPOT_HANDOFF.md
                     reference workflow + the paused DeePseudopot (neural pseudopotential) work
```

## Development

From a checkout of the repository:

```bash
uv sync                               # editable install + the dev group (Jupyter, anthropic, pytest)
pytest                                # ~15 s, no GPU or network: scripted model, QE mocked
pytest -m vllm                        # one real agent step against a running vLLM server
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
