# Benchmark results

Generated 2026-10-13 23:46 by `chem-llm-bench report` from `bench/index.jsonl`; do not edit by hand. Each run's folder under `bench/runs/` has its live progress, summary, the checks and what changed since the previous run of that compound.

## Suite `core`

The compounds developed against: rutile TiO2 (the reference benchmark) and the test34 sweep (polar wurtzite, III-V with a small gap, a semimetal, a heavy-element perovskite, a layered dichalcogenide).

| case | what it tests | last run | result | failed checks | gap eV (expected) | steps | wall min | prompt | commit | records |
|---|---|---|---|---|---|---|---|---|---|---|
| tio2 | reference benchmark; direct gap at Gamma, checked at every high-symmetry point | never | - | - | - (1.7-2.0) | - | - | - | - | - |
| zno | polar hexagonal cell, no inversion centre; Zn 3d semicore | 2026-10-07T00:26 | pass | - | 0.754 (0.4-1.1) | 54 | 42.64 | - | 8030a6a9 | [summary](runs/2026-10-13_234437_zno/summary.txt), [progress](runs/2026-10-13_234437_zno/progress.txt), [changes](runs/2026-10-13_234437_zno/changes.txt) |
| gaas | III-V with a small direct gap (PBE nearly closes it) | 2026-10-07T01:46 | pass | - | 0.077 (0.02-0.6) | 55 | 80.21 | - | 8030a6a9 | [summary](runs/2026-10-13_234439_gaas/summary.txt), [progress](runs/2026-10-13_234439_gaas/progress.txt), [changes](runs/2026-10-13_234439_gaas/changes.txt) |
| graphite | semimetal, needs smearing; layered hexagonal | 2026-10-07T02:18 | pass | - | -0.005 (metal) | 40 | 31.94 | - | 8030a6a9 | [summary](runs/2026-10-13_234439_graphite/summary.txt), [progress](runs/2026-10-13_234439_graphite/progress.txt), [changes](runs/2026-10-13_234439_graphite/changes.txt) |
| cspbbr3 | heavy elements, spin-orbit required (it shrinks the gap by ~1 eV) | 2026-10-07T03:17 | pass | - | 0.386 (0.2-0.9) | 58 | 59.15 | - | 8030a6a9 | [summary](runs/2026-10-13_234440_cspbbr3/summary.txt), [progress](runs/2026-10-13_234440_cspbbr3/progress.txt), [changes](runs/2026-10-13_234440_cspbbr3/changes.txt) |
| mos2 | layered dichalcogenide with an indirect gap | 2026-10-07T03:55 | pass | - | 1.203 (0.75-1.5) | 47 | 38.37 | - | 8030a6a9 | [summary](runs/2026-10-13_234440_mos2/summary.txt), [progress](runs/2026-10-13_234440_mos2/progress.txt), [changes](runs/2026-10-13_234440_mos2/changes.txt) |

## Suite `extended`

Compounds the prompt was never tuned on: elemental and ionic insulators, an indirect-gap semiconductor, a nitride with semicore d states, a perovskite oxide, a d-band metal and a ferromagnet.

| case | what it tests | last run | result | failed checks | gap eV (expected) | steps | wall min | prompt | commit | records |
|---|---|---|---|---|---|---|---|---|---|---|
| si | textbook indirect gap (CBM near X, not at a labelled point) | never | - | - | - (0.4-0.8) | - | - | - | - | - |
| diamond | wide indirect gap; degenerate valence pairs at X | never | - | - | - (3.8-4.5) | - | - | - | - | - |
| gan | nitride with Ga 3d semicore states; direct gap at Gamma | never | - | - | - (1.4-2.4) | - | - | - | - | - |
| nacl | ionic wide-gap insulator, alkali semicore | never | - | - | - (4.4-5.6) | - | - | - | - | - |
| mgo | ionic oxide, direct gap at Gamma | never | - | - | - (4.0-5.3) | - | - | - | - | - |
| srtio3 | perovskite oxide, indirect R-Gamma gap from Ti 3d conduction bands | never | - | - | - (1.4-2.3) | - | - | - | - | - |
| cu | d-band metal, needs smearing | never | - | - | - (metal) | - | - | - | - | - |
| fe | ferromagnetic metal; must be spin-polarized (~2.2 muB/atom) | never | - | - | - (metal) | - | - | - | - | - |

## Suite `all`

core + extended.

| case | what it tests | last run | result | failed checks | gap eV (expected) | steps | wall min | prompt | commit | records |
|---|---|---|---|---|---|---|---|---|---|---|
| tio2 | reference benchmark; direct gap at Gamma, checked at every high-symmetry point | never | - | - | - (1.7-2.0) | - | - | - | - | - |
| zno | polar hexagonal cell, no inversion centre; Zn 3d semicore | 2026-10-07T00:26 | pass | - | 0.754 (0.4-1.1) | 54 | 42.64 | - | 8030a6a9 | [summary](runs/2026-10-13_234437_zno/summary.txt), [progress](runs/2026-10-13_234437_zno/progress.txt), [changes](runs/2026-10-13_234437_zno/changes.txt) |
| gaas | III-V with a small direct gap (PBE nearly closes it) | 2026-10-07T01:46 | pass | - | 0.077 (0.02-0.6) | 55 | 80.21 | - | 8030a6a9 | [summary](runs/2026-10-13_234439_gaas/summary.txt), [progress](runs/2026-10-13_234439_gaas/progress.txt), [changes](runs/2026-10-13_234439_gaas/changes.txt) |
| graphite | semimetal, needs smearing; layered hexagonal | 2026-10-07T02:18 | pass | - | -0.005 (metal) | 40 | 31.94 | - | 8030a6a9 | [summary](runs/2026-10-13_234439_graphite/summary.txt), [progress](runs/2026-10-13_234439_graphite/progress.txt), [changes](runs/2026-10-13_234439_graphite/changes.txt) |
| cspbbr3 | heavy elements, spin-orbit required (it shrinks the gap by ~1 eV) | 2026-10-07T03:17 | pass | - | 0.386 (0.2-0.9) | 58 | 59.15 | - | 8030a6a9 | [summary](runs/2026-10-13_234440_cspbbr3/summary.txt), [progress](runs/2026-10-13_234440_cspbbr3/progress.txt), [changes](runs/2026-10-13_234440_cspbbr3/changes.txt) |
| mos2 | layered dichalcogenide with an indirect gap | 2026-10-07T03:55 | pass | - | 1.203 (0.75-1.5) | 47 | 38.37 | - | 8030a6a9 | [summary](runs/2026-10-13_234440_mos2/summary.txt), [progress](runs/2026-10-13_234440_mos2/progress.txt), [changes](runs/2026-10-13_234440_mos2/changes.txt) |
| si | textbook indirect gap (CBM near X, not at a labelled point) | never | - | - | - (0.4-0.8) | - | - | - | - | - |
| diamond | wide indirect gap; degenerate valence pairs at X | never | - | - | - (3.8-4.5) | - | - | - | - | - |
| gan | nitride with Ga 3d semicore states; direct gap at Gamma | never | - | - | - (1.4-2.4) | - | - | - | - | - |
| nacl | ionic wide-gap insulator, alkali semicore | never | - | - | - (4.4-5.6) | - | - | - | - | - |
| mgo | ionic oxide, direct gap at Gamma | never | - | - | - (4.0-5.3) | - | - | - | - | - |
| srtio3 | perovskite oxide, indirect R-Gamma gap from Ti 3d conduction bands | never | - | - | - (1.4-2.3) | - | - | - | - | - |
| cu | d-band metal, needs smearing | never | - | - | - (metal) | - | - | - | - | - |
| fe | ferromagnetic metal; must be spin-polarized (~2.2 muB/atom) | never | - | - | - (metal) | - | - | - | - | - |

## History

### zno: polar hexagonal cell, no inversion centre; Zn 3d semicore

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-07T00:26 | test34 rerun (in-process transformers, before the benchmark existed) | qwen | pass | - | 0.754 | 54 | 42.64 | - | - | 8030a6a9 |

### gaas: III-V with a small direct gap (PBE nearly closes it)

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-07T01:46 | test34 rerun (in-process transformers, before the benchmark existed) | qwen | pass | - | 0.077 | 55 | 80.21 | - | - | 8030a6a9 |

### graphite: semimetal, needs smearing; layered hexagonal

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-07T02:18 | test34 rerun (in-process transformers, before the benchmark existed) | qwen | pass | - | -0.005 | 40 | 31.94 | - | - | 8030a6a9 |

### cspbbr3: heavy elements, spin-orbit required (it shrinks the gap by ~1 eV)

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-07T03:17 | test34 rerun (in-process transformers, before the benchmark existed) | qwen | pass | - | 0.386 | 58 | 59.15 | - | - | 8030a6a9 |

### mos2: layered dichalcogenide with an indirect gap

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-07T03:55 | test34 rerun (in-process transformers, before the benchmark existed) | qwen | pass | - | 1.203 | 47 | 38.37 | - | - | 8030a6a9 |
