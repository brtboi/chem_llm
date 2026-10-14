# Benchmark results

Generated 2026-10-14 01:12 by `chem-llm-bench report` from `bench/index.jsonl`; do not edit by hand. Each run's folder under `bench/runs/` has its live progress, summary, the checks and what changed since the previous run of that compound.

## Suite `core`

The compounds developed against: rutile TiO2 (the reference benchmark) and the test34 sweep (polar wurtzite, III-V with a small gap, a semimetal, a heavy-element perovskite, a layered dichalcogenide).

| case | what it tests | last run | result | failed checks | gap eV (expected) | steps | wall min | prompt | commit | records |
|---|---|---|---|---|---|---|---|---|---|---|
| tio2 | reference benchmark; direct gap at Gamma, checked at every high-symmetry point | 2026-10-14T00:45 | pass | - | 1.730 (1.7-2.0) | 33 | 9.66 | d780f938 | e3df222f | [summary](runs/2026-10-14_003549_tio2/summary.txt), [progress](runs/2026-10-14_003549_tio2/progress.txt), [changes](runs/2026-10-14_003549_tio2/changes.txt) |
| zno | polar hexagonal cell, no inversion centre; Zn 3d semicore | 2026-10-14T00:53 | pass | - | 0.754 (0.4-1.1) | 39 | 17.42 | b86da611 | e3df222f | [summary](runs/2026-10-14_003549_zno/summary.txt), [progress](runs/2026-10-14_003549_zno/progress.txt), [changes](runs/2026-10-14_003549_zno/changes.txt) |
| gaas | III-V with a small direct gap (PBE nearly closes it) | 2026-10-14T00:29 | pass | - | 0.077 (0.02-0.6) | 48 | 42.94 | 1f0e056a | aa252971 | [summary](runs/2026-10-13_234637_gaas/summary.txt), [progress](runs/2026-10-13_234637_gaas/progress.txt), [changes](runs/2026-10-13_234637_gaas/changes.txt) |
| graphite | semimetal, needs smearing; layered hexagonal | 2026-10-14T01:02 | pass | - | -0.005 (metal) | 32 | 26.8 | 24c763aa | e3df222f | [summary](runs/2026-10-14_003549_graphite/summary.txt), [progress](runs/2026-10-14_003549_graphite/progress.txt), [changes](runs/2026-10-14_003549_graphite/changes.txt) |
| cspbbr3 | heavy elements, spin-orbit required (it shrinks the gap by ~1 eV) | 2026-10-14T00:22 | pass | - | 0.373 (0.2-0.9) | 46 | 36.3 | 5763242e | aa252971 | [summary](runs/2026-10-13_234637_cspbbr3/summary.txt), [progress](runs/2026-10-13_234637_cspbbr3/progress.txt), [changes](runs/2026-10-13_234637_cspbbr3/changes.txt) |
| mos2 | layered dichalcogenide with an indirect gap | 2026-10-14T00:54 | pass | - | 1.203 (0.75-1.5) | 42 | 18.67 | 1bc2d5b7 | e3df222f | [summary](runs/2026-10-14_003549_mos2/summary.txt), [progress](runs/2026-10-14_003549_mos2/progress.txt), [changes](runs/2026-10-14_003549_mos2/changes.txt) |

## Suite `extended`

Compounds the prompt was never tuned on: elemental and ionic insulators, an indirect-gap semiconductor, a nitride with semicore d states, a perovskite oxide, a d-band metal and a ferromagnet.

| case | what it tests | last run | result | failed checks | gap eV (expected) | steps | wall min | prompt | commit | records |
|---|---|---|---|---|---|---|---|---|---|---|
| si | textbook indirect gap (CBM near X, not at a labelled point) | 2026-10-14T00:44 | pass | - | 0.472 (0.4-0.8) | 34 | 8.74 | f8789fb5 | e3df222f | [summary](runs/2026-10-14_003549_si/summary.txt), [progress](runs/2026-10-14_003549_si/progress.txt), [changes](runs/2026-10-14_003549_si/changes.txt) |
| diamond | wide indirect gap; degenerate valence pairs at X | 2026-10-14T00:57 | pass | - | 4.055 (3.8-4.5) | 27 | 12.99 | e982a950 | e3df222f | [summary](runs/2026-10-14_004443_diamond/summary.txt), [progress](runs/2026-10-14_004443_diamond/progress.txt), [changes](runs/2026-10-14_004443_diamond/changes.txt) |
| gan | nitride with Ga 3d semicore states; direct gap at Gamma | 2026-10-14T01:07 | pass | - | 1.866 (1.4-2.4) | 36 | 21.4 | d86c1598 | e3df222f | [summary](runs/2026-10-14_004538_gan/summary.txt), [progress](runs/2026-10-14_004538_gan/progress.txt), [changes](runs/2026-10-14_004538_gan/changes.txt) |
| nacl | ionic wide-gap insulator, alkali semicore | 2026-10-14T01:05 | pass | - | 5.038 (4.4-5.6) | 31 | 12.48 | 078b8223 | e3df222f | [summary](runs/2026-10-14_005324_nacl/summary.txt), [progress](runs/2026-10-14_005324_nacl/progress.txt), [changes](runs/2026-10-14_005324_nacl/changes.txt) |
| mgo | ionic oxide, direct gap at Gamma | never | - | - | - (4.0-5.3) | - | - | - | - | - |
| srtio3 | perovskite oxide, indirect R-Gamma gap from Ti 3d conduction bands | never | - | - | - (1.4-2.3) | - | - | - | - | - |
| cu | d-band metal, needs smearing | never | - | - | - (metal) | - | - | - | - | - |
| fe | ferromagnetic metal; must be spin-polarized (~2.2 muB/atom) | never | - | - | - (metal) | - | - | - | - | - |

## Suite `all`

core + extended.

| case | what it tests | last run | result | failed checks | gap eV (expected) | steps | wall min | prompt | commit | records |
|---|---|---|---|---|---|---|---|---|---|---|
| tio2 | reference benchmark; direct gap at Gamma, checked at every high-symmetry point | 2026-10-14T00:45 | pass | - | 1.730 (1.7-2.0) | 33 | 9.66 | d780f938 | e3df222f | [summary](runs/2026-10-14_003549_tio2/summary.txt), [progress](runs/2026-10-14_003549_tio2/progress.txt), [changes](runs/2026-10-14_003549_tio2/changes.txt) |
| zno | polar hexagonal cell, no inversion centre; Zn 3d semicore | 2026-10-14T00:53 | pass | - | 0.754 (0.4-1.1) | 39 | 17.42 | b86da611 | e3df222f | [summary](runs/2026-10-14_003549_zno/summary.txt), [progress](runs/2026-10-14_003549_zno/progress.txt), [changes](runs/2026-10-14_003549_zno/changes.txt) |
| gaas | III-V with a small direct gap (PBE nearly closes it) | 2026-10-14T00:29 | pass | - | 0.077 (0.02-0.6) | 48 | 42.94 | 1f0e056a | aa252971 | [summary](runs/2026-10-13_234637_gaas/summary.txt), [progress](runs/2026-10-13_234637_gaas/progress.txt), [changes](runs/2026-10-13_234637_gaas/changes.txt) |
| graphite | semimetal, needs smearing; layered hexagonal | 2026-10-14T01:02 | pass | - | -0.005 (metal) | 32 | 26.8 | 24c763aa | e3df222f | [summary](runs/2026-10-14_003549_graphite/summary.txt), [progress](runs/2026-10-14_003549_graphite/progress.txt), [changes](runs/2026-10-14_003549_graphite/changes.txt) |
| cspbbr3 | heavy elements, spin-orbit required (it shrinks the gap by ~1 eV) | 2026-10-14T00:22 | pass | - | 0.373 (0.2-0.9) | 46 | 36.3 | 5763242e | aa252971 | [summary](runs/2026-10-13_234637_cspbbr3/summary.txt), [progress](runs/2026-10-13_234637_cspbbr3/progress.txt), [changes](runs/2026-10-13_234637_cspbbr3/changes.txt) |
| mos2 | layered dichalcogenide with an indirect gap | 2026-10-14T00:54 | pass | - | 1.203 (0.75-1.5) | 42 | 18.67 | 1bc2d5b7 | e3df222f | [summary](runs/2026-10-14_003549_mos2/summary.txt), [progress](runs/2026-10-14_003549_mos2/progress.txt), [changes](runs/2026-10-14_003549_mos2/changes.txt) |
| si | textbook indirect gap (CBM near X, not at a labelled point) | 2026-10-14T00:44 | pass | - | 0.472 (0.4-0.8) | 34 | 8.74 | f8789fb5 | e3df222f | [summary](runs/2026-10-14_003549_si/summary.txt), [progress](runs/2026-10-14_003549_si/progress.txt), [changes](runs/2026-10-14_003549_si/changes.txt) |
| diamond | wide indirect gap; degenerate valence pairs at X | 2026-10-14T00:57 | pass | - | 4.055 (3.8-4.5) | 27 | 12.99 | e982a950 | e3df222f | [summary](runs/2026-10-14_004443_diamond/summary.txt), [progress](runs/2026-10-14_004443_diamond/progress.txt), [changes](runs/2026-10-14_004443_diamond/changes.txt) |
| gan | nitride with Ga 3d semicore states; direct gap at Gamma | 2026-10-14T01:07 | pass | - | 1.866 (1.4-2.4) | 36 | 21.4 | d86c1598 | e3df222f | [summary](runs/2026-10-14_004538_gan/summary.txt), [progress](runs/2026-10-14_004538_gan/progress.txt), [changes](runs/2026-10-14_004538_gan/changes.txt) |
| nacl | ionic wide-gap insulator, alkali semicore | 2026-10-14T01:05 | pass | - | 5.038 (4.4-5.6) | 31 | 12.48 | 078b8223 | e3df222f | [summary](runs/2026-10-14_005324_nacl/summary.txt), [progress](runs/2026-10-14_005324_nacl/progress.txt), [changes](runs/2026-10-14_005324_nacl/changes.txt) |
| mgo | ionic oxide, direct gap at Gamma | never | - | - | - (4.0-5.3) | - | - | - | - | - |
| srtio3 | perovskite oxide, indirect R-Gamma gap from Ti 3d conduction bands | never | - | - | - (1.4-2.3) | - | - | - | - | - |
| cu | d-band metal, needs smearing | never | - | - | - (metal) | - | - | - | - | - |
| fe | ferromagnetic metal; must be spin-polarized (~2.2 muB/atom) | never | - | - | - (metal) | - | - | - | - | - |

## History

### zno: polar hexagonal cell, no inversion centre; Zn 3d semicore

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T00:53 | sweep 2: qe_output reader, tick lists, lsym off, 3 QE slots; extended baseline | vllm | pass | - | 0.754 | 39 | 17.42 | 5.52 | b86da611 | e3df222f |
| 2026-10-14T00:25 | sweep 1: vLLM + LangGraph, 1ebd4df plot fixes | vllm | pass | - | 0.754 | 47 | 38.65 | 6.23 | 5e9cf136 | aa252971 |
| 2026-10-07T00:26 | test34 rerun (in-process transformers, before the benchmark existed) | qwen | pass | - | 0.754 | 54 | 42.64 | - | - | 8030a6a9 |

### gaas: III-V with a small direct gap (PBE nearly closes it)

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T00:29 | sweep 1: vLLM + LangGraph, 1ebd4df plot fixes | vllm | pass | - | 0.077 | 48 | 42.94 | 8.92 | 1f0e056a | aa252971 |
| 2026-10-07T01:46 | test34 rerun (in-process transformers, before the benchmark existed) | qwen | pass | - | 0.077 | 55 | 80.21 | - | - | 8030a6a9 |

### graphite: semimetal, needs smearing; layered hexagonal

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T01:02 | sweep 2: qe_output reader, tick lists, lsym off, 3 QE slots; extended baseline | vllm | pass | - | -0.005 | 32 | 26.8 | 5.33 | 24c763aa | e3df222f |
| 2026-10-13T23:58 | sweep 1: vLLM + LangGraph, 1ebd4df plot fixes | vllm | pass | - | -0.005 | 50 | 12.33 | 7.06 | 21ba0266 | aa252971 |
| 2026-10-07T02:18 | test34 rerun (in-process transformers, before the benchmark existed) | qwen | pass | - | -0.005 | 40 | 31.94 | - | - | 8030a6a9 |

### cspbbr3: heavy elements, spin-orbit required (it shrinks the gap by ~1 eV)

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T00:22 | sweep 1: vLLM + LangGraph, 1ebd4df plot fixes | vllm | pass | - | 0.373 | 46 | 36.3 | 6.62 | 5763242e | aa252971 |
| 2026-10-07T03:17 | test34 rerun (in-process transformers, before the benchmark existed) | qwen | pass | - | 0.386 | 58 | 59.15 | - | - | 8030a6a9 |

### mos2: layered dichalcogenide with an indirect gap

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T00:54 | sweep 2: qe_output reader, tick lists, lsym off, 3 QE slots; extended baseline | vllm | pass | - | 1.203 | 42 | 18.67 | 6.58 | 1bc2d5b7 | e3df222f |
| 2026-10-14T00:31 | sweep 1: vLLM + LangGraph, 1ebd4df plot fixes | vllm | pass | - | 1.203 | 63 | 45.29 | 6.84 | 499c9704 | aa252971 |
| 2026-10-07T03:55 | test34 rerun (in-process transformers, before the benchmark existed) | qwen | pass | - | 1.203 | 47 | 38.37 | - | - | 8030a6a9 |

### tio2: reference benchmark; direct gap at Gamma, checked at every high-symmetry point

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T00:45 | sweep 2: qe_output reader, tick lists, lsym off, 3 QE slots; extended baseline | vllm | pass | - | 1.730 | 33 | 9.66 | 4.45 | d780f938 | e3df222f |
| 2026-10-13T23:54 | sweep 1: vLLM + LangGraph, 1ebd4df plot fixes | vllm | pass | - | 1.838 | 30 | 8.32 | 4.57 | 6868c3d7 | aa252971 |

### si: textbook indirect gap (CBM near X, not at a labelled point)

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T00:44 | sweep 2: qe_output reader, tick lists, lsym off, 3 QE slots; extended baseline | vllm | pass | - | 0.472 | 34 | 8.74 | 4.54 | f8789fb5 | e3df222f |

### diamond: wide indirect gap; degenerate valence pairs at X

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T00:57 | sweep 2: qe_output reader, tick lists, lsym off, 3 QE slots; extended baseline | vllm | pass | - | 4.055 | 27 | 12.99 | 5.14 | e982a950 | e3df222f |

### nacl: ionic wide-gap insulator, alkali semicore

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T01:05 | sweep 2: qe_output reader, tick lists, lsym off, 3 QE slots; extended baseline | vllm | pass | - | 5.038 | 31 | 12.48 | 4.93 | 078b8223 | e3df222f |

### gan: nitride with Ga 3d semicore states; direct gap at Gamma

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T01:07 | sweep 2: qe_output reader, tick lists, lsym off, 3 QE slots; extended baseline | vllm | pass | - | 1.866 | 36 | 21.4 | 5.31 | d86c1598 | e3df222f |
