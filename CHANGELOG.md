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
- *Sweep 2, all 14 compounds* (8 core + extended baseline): the 8 that finished all passed
  (Si, diamond, NaCl, GaN on first try). MoS2 took 42 steps, down from 63. I stopped it by
  hand: three QE jobs and vLLM time-slicing the same four GPUs made each job far slower than
  running alone (a 3-minute bands run made no progress in 24 min) (15c2b0a). It also showed:
  - TiO2 got a 1.730 eV gap from a hand-picked 4×4×2 SCF mesh that under-sampled the short
    c-axis. New `scf_kmesh` (even reciprocal-space spacing, denser for metals) and a rule
    requiring it.
  - Fe: the agent ran non-magnetic with no prompt cue. `generate_cif` now returns Materials
    Project's magnetic ordering and moment, and a magnetism rule acts on it.
- *Infrastructure:* QE now gets dedicated GPUs (vLLM on GPUs 0–1 with TP=2, one QE job per
  GPU on 2–3). Sweep 3 showed the pin wasn't working, because `srun` resets
  `CUDA_VISIBLE_DEVICES` per step; it is now set inside the launch. A silicon SCF went from
  4 min contended to 3 s. Agent processes no longer create a stray 700 MB CUDA context.
  Protocol rule 11: QE runs only through the tool (two agents had started a `find /global`
  for pw.x) (1b8c79f).
- *Sweep 4, all 14:* **13/14 pass, the whole suite in about 23 min**. TiO2 is back to
  1.848 eV, with every high-symmetry point within 0.007 eV of the references. Golden
  references were promoted for the 13, each checked at its high-symmetry points:
  - degenerate pairs at X in Si and diamond;
  - SrTiO3's indirect R→Γ gap;
  - Kramers pairs in CsPbBr3 with spin-orbit;
  - Cu d bands at −5.4 to −1.5 eV.

  Fe failed: the agent put `starting_magnetization` in `input_data`, and ASE's per-species
  value, taken from the atoms' zero moments, overrode it. The rule now says to set moments on
  the cell, and `validate_qe_input` flags spin-polarized inputs with zero moments (61135fe).
- *Sweep 5, all 14:* **14/14 pass in about 20 min**, each checked against its golden
  reference. Fe is now spin-polarized at 2.25 μB/atom (PBE ≈ 2.2–2.3; experiment 2.22).
  Run archives also keep the agent's final scripts and band plot (0e1f853).
- *Sweep 6, 21 cases:* the 14 pass again. A new **stretch suite** of 7 compounds never used
  for tuning passed 5 on the first attempt:
  - Al: band bottom at −11.1 eV;
  - CdTe: 0.549 eV, direct at Γ;
  - LiF: 8.59 eV;
  - PbTe: 0.054 eV with spin-orbit, direct at L;
  - Bi2Se3: 0.122 eV with spin-orbit, camel-back valence band near Γ (band inversion).

  The two failures (5f94842):
  - Ni ran non-magnetic. Materials Project reports fcc Ni as "not magnetic, ordering
    Unknown" but with a 0.71 μB moment, so the rule now decides on the moment.
  - hBN missed my expected range, which was taken from literature for other stackings.
    It is now anchored to Materials Project's 3.97 eV for the same structure, with the
    correction documented.

  Also found and fixed my own bug: a misplaced decorator had registered a helper as
  `validate_qe_input`, so validation crashed all sweep. A test now checks every tool's
  registration.
- *Sweep 7, 21 cases:* **21/21 pass**. Ni is now spin-polarized at 0.61 μB/atom (PBE ≈ 0.62,
  experiment ≈ 0.6). hBN gave exactly the same 3.712 eV again. Every case now has a golden
  reference. Median run: 34 steps, 10.4 min; the slowest is Bi2Se3 with spin-orbit at
  29 min. The whole 21-compound sweep took 48 min, 8 at a time.

| sweep | cases | passed | median steps | median wall (min) |
|---|---|---|---|---|
| test34 rerun (in-process transformers, before this week) | 5 | 5 | 54 | 42.6 |
| 1: vLLM + LangGraph | 6 | 6 | 48 | 38.6 |
| 2: tool fixes (stopped early) | 8 of 14 finished | 8 | 34 | 17.4 |
| 4: QE on dedicated GPUs, k-mesh rule | 14 | 13 | 35 | 9.4 |
| 5: magnetization fix | 14 | 14 | 38 | 10.2 |
| 6: + 7 never-tuned compounds | 21 | 19 | 37 | 10.9 |
| 7: Ni moment rule, validator fix | 21 | 21 | 34 | 10.4 |
| 8: repeat, no changes | 21 | 20 (CsPbBr3 fixed and rerun: pass) | — | — |

- *Sweep 8 (repeat with no code changes):* 20/21 passed (Bi2Se3 0.127 eV, finishing just before the allocation ended). CsPbBr3 failed: the agent read the relativistic
  rule's "Z > 54 OR spin-orbit physics" as AND, and ran without spin-orbit (1.601 eV). The
  rule now states the two conditions separately, and `validate_qe_input` rejects any element
  past Xe without `lspinorb`. The rerun chose fully relativistic pseudopotentials from the
  start and passed at 0.390 eV (0cebfb2). This is the first failure that came purely from
  sampling variance, which is why rules that matter now also get a deterministic check.

**Still open**
- The extended and stretch suites were run on the same prompt they are scored with. A
  fresh set of held-out compounds would be a fairer test of generalization.
- Expected gap ranges are hand-set (one was wrong: hBN). Deriving them from Materials
  Project's gap for the selected structure would be more principled.
- Each sweep runs the same 14–21 compounds once. With sampling at temperature 0.3, the pass
  rate over several repeats is the honest number. Sweep 8, an unchanged repeat, found one
  sampling failure in 20; it is now caught by a validator check.
- Spin-polarized band plots show the majority spin only.
- No LangSmith experiments yet: the integration is built and tested offline, but it needs
  a `LANGSMITH_API_KEY`.

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
