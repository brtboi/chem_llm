# Changelog

A weekly record of progress: what changed and what the runs showed. The newest week comes
first. Commit hashes link the work to `git log`. Per-run detail, including results and what
changed since the previous run of each compound, is in [`bench/RESULTS.md`](bench/RESULTS.md)
and `bench/runs/`.

## Week of 2026-10-12

**Speed: the model, not Quantum ESPRESSO, was the bottleneck.** Measured on the test34 runs,
QE took about 2 of ZnO's 42 minutes, and it already ran on all four GPUs (one A100 per MPI
rank, verified by PCI bus ID). The other 40 minutes were Qwen generating at about 50 s per
step through in-process `transformers`.
- New `vllm` backend (`scripts/serve_vllm.sh`). It serves the same Qwen3.8-27B weights with
  paged attention and prefix caching, so the ~8K-token system prompt is computed once.
  Rutile TiO2 now runs end to end in **5.6 min, down from 45–60**, with every high-symmetry
  point within 0.009 eV of the reference runs (c161b28).
- The agent loop is now a **LangGraph** graph (`think -> act`). Every backend (transformers,
  vLLM, Claude) is a **LangChain** chat model, and any chat model can be plugged in with
  `DFTAgent(llm=...)`. LangSmith tracing turns on with `LANGSMITH_TRACING` (c161b28).
- **Run telemetry:** each step records model seconds, tool seconds and tokens. Each run logs
  time per phase (model / QE / scripts / docs), per QE stage and per workflow STEP. A live
  `trace.jsonl` survives a run killed by an expired allocation. The run summary gained
  section 5, "Where the time went" (c161b28).
- **Prompt versioning:** each run records its full system prompt and a version hash.
  `chem-llm --diff-runs A B` compares two runs: outcome, cost, tool use, settings and a
  line-by-line prompt diff (c161b28).
- **Test suite:** `pytest` runs the real agent loop with a scripted model and mocked QE, with
  no GPU or network, in about 15 s (ff4beb1).
- **Run archive:** every run, from the CLI, the notebook or the benchmark, now keeps a
  permanent folder with its readable progress stream, summary, metrics and a
  `changes.txt` stating what changed since the previous run of that compound: commits in
  between, prompt diff, settings, result deltas. The folder is never cleared.
- **Benchmark framework** (`chem-llm-bench`): one suite file (`bench/suites.yaml`) lists every
  compound, what it tests and the physics expected (gap range or metal, spin-orbit,
  magnetism). Cases run in parallel against one vLLM server, with QE taking turns on the
  GPUs. They are scored automatically from the files each run produced, and the history is
  kept in `bench/index.jsonl` and `bench/RESULTS.md`. This replaces numbered sandbox
  folders. `--langsmith` also records each sweep as a LangSmith experiment.
- CLI flags renamed: `--pretty-progress` (live stream), `--summarize` (summary at the end),
  `--summarize-log` (past run) (1299a57).

**Benchmark sweeps (this week)**
- *Sweep 1, core suite, 6 compounds in parallel:* **6/6 pass in 45 min**. The same five
  test34 compounds took 252 min one at a time on in-process transformers. TiO2 1.838 eV,
  with every high-symmetry point within 0.009 eV of the reference runs; it is now the golden
  reference. ZnO 0.754, GaAs 0.077, graphite semimetal, CsPbBr3 0.373 with spin-orbit,
  MoS2 1.203 (e40bd1c).
- *What sweep 1 exposed, and the fixes* (e3df222):
  - ZnO, GaAs and MoS2 each spent ~30 of their ~40 min waiting for the single QE slot,
    mostly behind 20 min of bands.x symmetry analysis for CsPbBr3. Fixes: `lsym=.false.`,
    and up to 3 QE jobs at once (`srun --overlap`).
  - 5 of 6 plot scripts mis-unpacked the tick list. `plot_axis` now also returns plain
    `tick_x`/`tick_labels`.
  - Hand-written `pw.out` regexes failed. New `chem_llm.qe_output.read_pw_out` /
    `band_edges`.
  - Up to 7 failed attempts on invented spglib APIs for a space-group check. The rule now
    points to the space group the k-path tool already returns.

## Week of 2026-10-05

- `generate_cif` picks the most stable Materials Project structure, not the first one
  returned (3d37d16 era).
- **test34:** a five-compound sweep (graphite, CsPbBr3, GaAs, MoS2, ZnO). The DFT was correct
  for all five, but the plots and labels had problems, and ZnO ran out of steps after
  rewriting the same script five times.
- **k-path tool** (`chem_llm.kpath`): one standard seekpath path shared by the QE input and the
  plot, so the ticks always match the computed k-points.
- Prompt: k-path via the tool, fewer notes, an occupations rule (smearing for metals). A
  harness guard refuses no-op file rewrites.
- Sampling at temperature 0.3 instead of greedy decoding, so a stuck agent can break out of
  a loop.
- **test34 rerun:** all five compounds complete (ZnO 0.754 eV, GaAs 0.077, graphite semimetal,
  CsPbBr3 0.386 with spin-orbit, MoS2 1.203). Total time fell from 325 to 252 min. Plot
  connectors across k-path breaks were fixed afterwards (1ebd4df).

## Week of 2026-09-28

- **Context management:** `trim_history` drops dead material from the prompt (superseded
  files, resolved failures, duplicate notes), cutting the median prompt by about 19%.
  `read_file` can read a line range.
- **Harness guards:** refuse back-to-back notes and repeated identical doc searches.
- `get_pseudopotential` reports facts parsed from the UPF file itself (valence, relativistic
  treatment, functional, recommended cutoff).
- Rutile TiO2 benchmark records (test29) and a checker comparing every high-symmetry point
  (`sandbox/check_tio2.py`).
- CsPbBr3 completes with spin-orbit coupling (test31). TiO2 regression-checked (test32).
- **Quantum ESPRESSO on the GPUs:** about 12x faster SCF and 34x faster bands for TiO2.
- **Run summary:** four sections, each written for its own reader (steps, DFT setup for
  physicists, agent/harness for engineers, errors and doubts). Runs record their
  configuration. A readable live progress stream was added.

## Week of 2026-09-21

- ASE migration complete. Claude API backend added as an opt-in alternative.
- `validate_qe_input` catches misplaced QE namelist variables. Doc lookup is forced on a
  first failure.
- **Packaging:** config-driven, `DFTAgent` with per-agent `Settings`, published on PyPI as
  `chem-llm` 0.1.0.
- Fixed a cwd leak between runs, the occupied-band count derivation, and k-path derivation
  (seekpath's primitive cell, no reprojection).

## Weeks of 2026-08-31 and 2026-09-07

- Structure and QE input pipeline migrated from pymatgen to ASE.
- System prompt restructured into AGENT PROTOCOL / INVARIANTS / DECISION RULES.
- Optional DeePseudopot (neural pseudopotential) example wired in, now paused.

## Week of 2026-08-24

- TiO2 (test16) and CsPbBr3 (test17) runs. Replay a logged run's tool calls without the model.

## Weeks of 2026-08-03 and 2026-08-10

- Pseudopotential tool (pseudohub / Pseudo-Dojo).
- **Documentation search:** a scraped ASE/QE doc index with hybrid BM25 + vector retrieval,
  with the prompt requiring lookups.
- Work-dir management (`clear_dir`, `make_dir`), and remembered tool calls that survive
  history truncation.
- **First end-to-end successful run** (2026-08-12).

## Weeks of 2026-07-13 to 2026-07-27

- Structures from the Materials Project (replacing CrystaLLM generation). Agent logging.
- Tools accept a space group symbol and/or number. The agent generates perturbed TiO2.
- First pseudopotential tool (scalar-relativistic PBE from Pseudo-Dojo).

## Weeks of 2026-06-22 to 2026-07-06

- Project setup (uv), prototype agent pipeline, tool-call parsing tests.
- Stateful sequential tool-calling agent. Local Qwen3 30B model. CrystaLLM experiments for
  structure generation.
