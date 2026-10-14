# Benchmark results

Generated 2026-10-14 03:23 by `chem-llm-bench report` from `bench/index.jsonl`; do not edit by hand. Each run's folder under `bench/runs/` has its live progress, summary, the checks and what changed since the previous run of that compound.

## Suite `core`

The compounds developed against: rutile TiO2 (the reference benchmark) and the test34 sweep (polar wurtzite, III-V with a small gap, a semimetal, a heavy-element perovskite, a layered dichalcogenide).

| case | what it tests | last run | result | failed checks | gap eV (expected) | steps | wall min | prompt | commit | records |
|---|---|---|---|---|---|---|---|---|---|---|
| tio2 | reference benchmark; direct gap at Gamma, checked at every high-symmetry point | 2026-10-14T02:44 | pass | - | 1.848 (1.7-2.0) | 35 | 10.78 | 3861f7c7 | 0e1f853c | [summary](runs/2026-10-14_023337_tio2/summary.txt), [progress](runs/2026-10-14_023337_tio2/progress.txt), [changes](runs/2026-10-14_023337_tio2/changes.txt) |
| zno | polar hexagonal cell, no inversion centre; Zn 3d semicore | 2026-10-14T02:49 | pass | - | 0.755 (0.4-1.1) | 45 | 15.78 | 4e6cf1a5 | 0e1f853c | [summary](runs/2026-10-14_023337_zno/summary.txt), [progress](runs/2026-10-14_023337_zno/progress.txt), [changes](runs/2026-10-14_023337_zno/changes.txt) |
| gaas | III-V with a small direct gap (PBE nearly closes it) | 2026-10-14T02:44 | pass | - | 0.075 (0.02-0.6) | 34 | 10.48 | 77d6e749 | 0e1f853c | [summary](runs/2026-10-14_023337_gaas/summary.txt), [progress](runs/2026-10-14_023337_gaas/progress.txt), [changes](runs/2026-10-14_023337_gaas/changes.txt) |
| graphite | semimetal, needs smearing; layered hexagonal | 2026-10-14T02:49 | pass | - | -0.005 (metal) | 61 | 15.69 | 843005b9 | 0e1f853c | [summary](runs/2026-10-14_023337_graphite/summary.txt), [progress](runs/2026-10-14_023337_graphite/progress.txt), [changes](runs/2026-10-14_023337_graphite/changes.txt) |
| cspbbr3 | heavy elements, spin-orbit required (it shrinks the gap by ~1 eV) | 2026-10-14T02:49 | pass | - | 0.390 (0.2-0.9) | 43 | 15.95 | 2d3731e5 | 0e1f853c | [summary](runs/2026-10-14_023337_cspbbr3/summary.txt), [progress](runs/2026-10-14_023337_cspbbr3/progress.txt), [changes](runs/2026-10-14_023337_cspbbr3/changes.txt) |
| mos2 | layered dichalcogenide with an indirect gap | 2026-10-14T02:44 | pass | - | 1.203 (0.75-1.5) | 42 | 10.87 | abdf34dc | 0e1f853c | [summary](runs/2026-10-14_023337_mos2/summary.txt), [progress](runs/2026-10-14_023337_mos2/progress.txt), [changes](runs/2026-10-14_023337_mos2/changes.txt) |

## Suite `extended`

Compounds the prompt was never tuned on: elemental and ionic insulators, an indirect-gap semiconductor, a nitride with semicore d states, a perovskite oxide, a d-band metal and a ferromagnet.

| case | what it tests | last run | result | failed checks | gap eV (expected) | steps | wall min | prompt | commit | records |
|---|---|---|---|---|---|---|---|---|---|---|
| si | textbook indirect gap (CBM near X, not at a labelled point) | 2026-10-14T02:44 | pass | - | 0.472 (0.4-0.8) | 39 | 10.88 | 6e0d58cc | 0e1f853c | [summary](runs/2026-10-14_023337_si/summary.txt), [progress](runs/2026-10-14_023337_si/progress.txt), [changes](runs/2026-10-14_023337_si/changes.txt) |
| diamond | wide indirect gap; degenerate valence pairs at X | 2026-10-14T02:49 | pass | - | 4.055 (3.8-4.5) | 41 | 15.41 | ff2577f2 | 0e1f853c | [summary](runs/2026-10-14_023337_diamond/summary.txt), [progress](runs/2026-10-14_023337_diamond/progress.txt), [changes](runs/2026-10-14_023337_diamond/changes.txt) |
| gan | nitride with Ga 3d semicore states; direct gap at Gamma | 2026-10-14T02:56 | pass | - | 1.877 (1.4-2.4) | 42 | 12.13 | 498c9c97 | 0e1f853c | [summary](runs/2026-10-14_024416_gan/summary.txt), [progress](runs/2026-10-14_024416_gan/progress.txt), [changes](runs/2026-10-14_024416_gan/changes.txt) |
| nacl | ionic wide-gap insulator, alkali semicore | 2026-10-14T02:53 | pass | - | 5.039 (4.4-5.6) | 34 | 8.55 | 28568d74 | 0e1f853c | [summary](runs/2026-10-14_024435_nacl/summary.txt), [progress](runs/2026-10-14_024435_nacl/progress.txt), [changes](runs/2026-10-14_024435_nacl/changes.txt) |
| mgo | ionic oxide, direct gap at Gamma | 2026-10-14T02:54 | pass | - | 4.782 (4.0-5.3) | 37 | 10.24 | af9867f2 | 0e1f853c | [summary](runs/2026-10-14_024441_mgo/summary.txt), [progress](runs/2026-10-14_024441_mgo/progress.txt), [changes](runs/2026-10-14_024441_mgo/changes.txt) |
| srtio3 | perovskite oxide, indirect R-Gamma gap from Ti 3d conduction bands | 2026-10-14T02:55 | pass | - | 1.839 (1.4-2.3) | 36 | 10.53 | 2a95e512 | 0e1f853c | [summary](runs/2026-10-14_024441_srtio3/summary.txt), [progress](runs/2026-10-14_024441_srtio3/progress.txt), [changes](runs/2026-10-14_024441_srtio3/changes.txt) |
| cu | d-band metal, needs smearing | 2026-10-14T02:57 | pass | - | -2.366 (metal) | 31 | 8.47 | 9957ab83 | 0e1f853c | [summary](runs/2026-10-14_024910_cu/summary.txt), [progress](runs/2026-10-14_024910_cu/progress.txt), [changes](runs/2026-10-14_024910_cu/changes.txt) |
| fe | ferromagnetic metal; must be spin-polarized (~2.2 muB/atom) | 2026-10-14T03:09 | pass | - | -1.868 (metal) | 47 | 20.25 | ee656768 | 0e1f853c | [summary](runs/2026-10-14_024927_fe/summary.txt), [progress](runs/2026-10-14_024927_fe/progress.txt), [changes](runs/2026-10-14_024927_fe/changes.txt) |

## Suite `stretch`

Added after the core and extended suites passed, never used for tuning: a simple metal, a weak ferromagnet, a layered wide-gap insulator, the widest-gap ionic crystal, a II-VI semiconductor, and two heavy-element spin-orbit compounds (a rock-salt narrow-gap and a layered topological insulator).

| case | what it tests | last run | result | failed checks | gap eV (expected) | steps | wall min | prompt | commit | records |
|---|---|---|---|---|---|---|---|---|---|---|
| al | free-electron-like sp metal, needs smearing | 2026-10-14T02:58 | pass | - | -13.061 (metal) | 31 | 9.03 | 026f9af5 | 0e1f853c | [summary](runs/2026-10-14_024933_al/summary.txt), [progress](runs/2026-10-14_024933_al/progress.txt), [changes](runs/2026-10-14_024933_al/changes.txt) |
| ni | weak itinerant ferromagnet (~0.6 muB/atom) | 2026-10-14T02:59 | **FAIL** | magnetization | -1.002 (metal) | 34 | 10.18 | 1095dd8f | 0e1f853c | [summary](runs/2026-10-14_024946_ni/summary.txt), [progress](runs/2026-10-14_024946_ni/progress.txt), [changes](runs/2026-10-14_024946_ni/changes.txt) |
| hbn | layered wide-gap insulator (same space group as graphite, not a metal) | 2026-10-14T03:02 | **FAIL** | band_gap | 3.712 (3.4-4.6) | 37 | 9.19 | 9d58e85a | 0e1f853c | [summary](runs/2026-10-14_025316_hbn/summary.txt), [progress](runs/2026-10-14_025316_hbn/progress.txt), [changes](runs/2026-10-14_025316_hbn/changes.txt) |
| lif | widest-gap ionic crystal; light atoms, Li 1s semicore | 2026-10-14T03:06 | pass | - | 8.589 (8.0-9.8) | 39 | 11.67 | a80c617d | 0e1f853c | [summary](runs/2026-10-14_025504_lif/summary.txt), [progress](runs/2026-10-14_025504_lif/progress.txt), [changes](runs/2026-10-14_025504_lif/changes.txt) |
| cdte | II-VI semiconductor with Cd 4d states; spin-orbit optional (Te Z=52) | 2026-10-14T03:05 | pass | - | 0.549 (0.1-1.1) | 34 | 10.0 | 81728f68 | 0e1f853c | [summary](runs/2026-10-14_025522_cdte/summary.txt), [progress](runs/2026-10-14_025522_cdte/progress.txt), [changes](runs/2026-10-14_025522_cdte/changes.txt) |
| pbte | heavy-element narrow gap at L; spin-orbit required | 2026-10-14T03:08 | pass | - | 0.054 (0.02-0.7) | 36 | 12.22 | 25fa0bbd | 0e1f853c | [summary](runs/2026-10-14_025632_pbte/summary.txt), [progress](runs/2026-10-14_025632_pbte/progress.txt), [changes](runs/2026-10-14_025632_pbte/changes.txt) |
| bi2se3 | layered topological insulator; spin-orbit required (inverts the gap at Gamma) | 2026-10-14T03:23 | pass | - | 0.123 (0.1-0.6) | 41 | 25.45 | 892ab881 | 0e1f853c | [summary](runs/2026-10-14_025748_bi2se3/summary.txt), [progress](runs/2026-10-14_025748_bi2se3/progress.txt), [changes](runs/2026-10-14_025748_bi2se3/changes.txt) |

## Suite `all`

core + extended.

| case | what it tests | last run | result | failed checks | gap eV (expected) | steps | wall min | prompt | commit | records |
|---|---|---|---|---|---|---|---|---|---|---|
| tio2 | reference benchmark; direct gap at Gamma, checked at every high-symmetry point | 2026-10-14T02:44 | pass | - | 1.848 (1.7-2.0) | 35 | 10.78 | 3861f7c7 | 0e1f853c | [summary](runs/2026-10-14_023337_tio2/summary.txt), [progress](runs/2026-10-14_023337_tio2/progress.txt), [changes](runs/2026-10-14_023337_tio2/changes.txt) |
| zno | polar hexagonal cell, no inversion centre; Zn 3d semicore | 2026-10-14T02:49 | pass | - | 0.755 (0.4-1.1) | 45 | 15.78 | 4e6cf1a5 | 0e1f853c | [summary](runs/2026-10-14_023337_zno/summary.txt), [progress](runs/2026-10-14_023337_zno/progress.txt), [changes](runs/2026-10-14_023337_zno/changes.txt) |
| gaas | III-V with a small direct gap (PBE nearly closes it) | 2026-10-14T02:44 | pass | - | 0.075 (0.02-0.6) | 34 | 10.48 | 77d6e749 | 0e1f853c | [summary](runs/2026-10-14_023337_gaas/summary.txt), [progress](runs/2026-10-14_023337_gaas/progress.txt), [changes](runs/2026-10-14_023337_gaas/changes.txt) |
| graphite | semimetal, needs smearing; layered hexagonal | 2026-10-14T02:49 | pass | - | -0.005 (metal) | 61 | 15.69 | 843005b9 | 0e1f853c | [summary](runs/2026-10-14_023337_graphite/summary.txt), [progress](runs/2026-10-14_023337_graphite/progress.txt), [changes](runs/2026-10-14_023337_graphite/changes.txt) |
| cspbbr3 | heavy elements, spin-orbit required (it shrinks the gap by ~1 eV) | 2026-10-14T02:49 | pass | - | 0.390 (0.2-0.9) | 43 | 15.95 | 2d3731e5 | 0e1f853c | [summary](runs/2026-10-14_023337_cspbbr3/summary.txt), [progress](runs/2026-10-14_023337_cspbbr3/progress.txt), [changes](runs/2026-10-14_023337_cspbbr3/changes.txt) |
| mos2 | layered dichalcogenide with an indirect gap | 2026-10-14T02:44 | pass | - | 1.203 (0.75-1.5) | 42 | 10.87 | abdf34dc | 0e1f853c | [summary](runs/2026-10-14_023337_mos2/summary.txt), [progress](runs/2026-10-14_023337_mos2/progress.txt), [changes](runs/2026-10-14_023337_mos2/changes.txt) |
| si | textbook indirect gap (CBM near X, not at a labelled point) | 2026-10-14T02:44 | pass | - | 0.472 (0.4-0.8) | 39 | 10.88 | 6e0d58cc | 0e1f853c | [summary](runs/2026-10-14_023337_si/summary.txt), [progress](runs/2026-10-14_023337_si/progress.txt), [changes](runs/2026-10-14_023337_si/changes.txt) |
| diamond | wide indirect gap; degenerate valence pairs at X | 2026-10-14T02:49 | pass | - | 4.055 (3.8-4.5) | 41 | 15.41 | ff2577f2 | 0e1f853c | [summary](runs/2026-10-14_023337_diamond/summary.txt), [progress](runs/2026-10-14_023337_diamond/progress.txt), [changes](runs/2026-10-14_023337_diamond/changes.txt) |
| gan | nitride with Ga 3d semicore states; direct gap at Gamma | 2026-10-14T02:56 | pass | - | 1.877 (1.4-2.4) | 42 | 12.13 | 498c9c97 | 0e1f853c | [summary](runs/2026-10-14_024416_gan/summary.txt), [progress](runs/2026-10-14_024416_gan/progress.txt), [changes](runs/2026-10-14_024416_gan/changes.txt) |
| nacl | ionic wide-gap insulator, alkali semicore | 2026-10-14T02:53 | pass | - | 5.039 (4.4-5.6) | 34 | 8.55 | 28568d74 | 0e1f853c | [summary](runs/2026-10-14_024435_nacl/summary.txt), [progress](runs/2026-10-14_024435_nacl/progress.txt), [changes](runs/2026-10-14_024435_nacl/changes.txt) |
| mgo | ionic oxide, direct gap at Gamma | 2026-10-14T02:54 | pass | - | 4.782 (4.0-5.3) | 37 | 10.24 | af9867f2 | 0e1f853c | [summary](runs/2026-10-14_024441_mgo/summary.txt), [progress](runs/2026-10-14_024441_mgo/progress.txt), [changes](runs/2026-10-14_024441_mgo/changes.txt) |
| srtio3 | perovskite oxide, indirect R-Gamma gap from Ti 3d conduction bands | 2026-10-14T02:55 | pass | - | 1.839 (1.4-2.3) | 36 | 10.53 | 2a95e512 | 0e1f853c | [summary](runs/2026-10-14_024441_srtio3/summary.txt), [progress](runs/2026-10-14_024441_srtio3/progress.txt), [changes](runs/2026-10-14_024441_srtio3/changes.txt) |
| cu | d-band metal, needs smearing | 2026-10-14T02:57 | pass | - | -2.366 (metal) | 31 | 8.47 | 9957ab83 | 0e1f853c | [summary](runs/2026-10-14_024910_cu/summary.txt), [progress](runs/2026-10-14_024910_cu/progress.txt), [changes](runs/2026-10-14_024910_cu/changes.txt) |
| fe | ferromagnetic metal; must be spin-polarized (~2.2 muB/atom) | 2026-10-14T03:09 | pass | - | -1.868 (metal) | 47 | 20.25 | ee656768 | 0e1f853c | [summary](runs/2026-10-14_024927_fe/summary.txt), [progress](runs/2026-10-14_024927_fe/progress.txt), [changes](runs/2026-10-14_024927_fe/changes.txt) |

## Suite `everything`

core + extended + stretch.

| case | what it tests | last run | result | failed checks | gap eV (expected) | steps | wall min | prompt | commit | records |
|---|---|---|---|---|---|---|---|---|---|---|
| tio2 | reference benchmark; direct gap at Gamma, checked at every high-symmetry point | 2026-10-14T02:44 | pass | - | 1.848 (1.7-2.0) | 35 | 10.78 | 3861f7c7 | 0e1f853c | [summary](runs/2026-10-14_023337_tio2/summary.txt), [progress](runs/2026-10-14_023337_tio2/progress.txt), [changes](runs/2026-10-14_023337_tio2/changes.txt) |
| zno | polar hexagonal cell, no inversion centre; Zn 3d semicore | 2026-10-14T02:49 | pass | - | 0.755 (0.4-1.1) | 45 | 15.78 | 4e6cf1a5 | 0e1f853c | [summary](runs/2026-10-14_023337_zno/summary.txt), [progress](runs/2026-10-14_023337_zno/progress.txt), [changes](runs/2026-10-14_023337_zno/changes.txt) |
| gaas | III-V with a small direct gap (PBE nearly closes it) | 2026-10-14T02:44 | pass | - | 0.075 (0.02-0.6) | 34 | 10.48 | 77d6e749 | 0e1f853c | [summary](runs/2026-10-14_023337_gaas/summary.txt), [progress](runs/2026-10-14_023337_gaas/progress.txt), [changes](runs/2026-10-14_023337_gaas/changes.txt) |
| graphite | semimetal, needs smearing; layered hexagonal | 2026-10-14T02:49 | pass | - | -0.005 (metal) | 61 | 15.69 | 843005b9 | 0e1f853c | [summary](runs/2026-10-14_023337_graphite/summary.txt), [progress](runs/2026-10-14_023337_graphite/progress.txt), [changes](runs/2026-10-14_023337_graphite/changes.txt) |
| cspbbr3 | heavy elements, spin-orbit required (it shrinks the gap by ~1 eV) | 2026-10-14T02:49 | pass | - | 0.390 (0.2-0.9) | 43 | 15.95 | 2d3731e5 | 0e1f853c | [summary](runs/2026-10-14_023337_cspbbr3/summary.txt), [progress](runs/2026-10-14_023337_cspbbr3/progress.txt), [changes](runs/2026-10-14_023337_cspbbr3/changes.txt) |
| mos2 | layered dichalcogenide with an indirect gap | 2026-10-14T02:44 | pass | - | 1.203 (0.75-1.5) | 42 | 10.87 | abdf34dc | 0e1f853c | [summary](runs/2026-10-14_023337_mos2/summary.txt), [progress](runs/2026-10-14_023337_mos2/progress.txt), [changes](runs/2026-10-14_023337_mos2/changes.txt) |
| si | textbook indirect gap (CBM near X, not at a labelled point) | 2026-10-14T02:44 | pass | - | 0.472 (0.4-0.8) | 39 | 10.88 | 6e0d58cc | 0e1f853c | [summary](runs/2026-10-14_023337_si/summary.txt), [progress](runs/2026-10-14_023337_si/progress.txt), [changes](runs/2026-10-14_023337_si/changes.txt) |
| diamond | wide indirect gap; degenerate valence pairs at X | 2026-10-14T02:49 | pass | - | 4.055 (3.8-4.5) | 41 | 15.41 | ff2577f2 | 0e1f853c | [summary](runs/2026-10-14_023337_diamond/summary.txt), [progress](runs/2026-10-14_023337_diamond/progress.txt), [changes](runs/2026-10-14_023337_diamond/changes.txt) |
| gan | nitride with Ga 3d semicore states; direct gap at Gamma | 2026-10-14T02:56 | pass | - | 1.877 (1.4-2.4) | 42 | 12.13 | 498c9c97 | 0e1f853c | [summary](runs/2026-10-14_024416_gan/summary.txt), [progress](runs/2026-10-14_024416_gan/progress.txt), [changes](runs/2026-10-14_024416_gan/changes.txt) |
| nacl | ionic wide-gap insulator, alkali semicore | 2026-10-14T02:53 | pass | - | 5.039 (4.4-5.6) | 34 | 8.55 | 28568d74 | 0e1f853c | [summary](runs/2026-10-14_024435_nacl/summary.txt), [progress](runs/2026-10-14_024435_nacl/progress.txt), [changes](runs/2026-10-14_024435_nacl/changes.txt) |
| mgo | ionic oxide, direct gap at Gamma | 2026-10-14T02:54 | pass | - | 4.782 (4.0-5.3) | 37 | 10.24 | af9867f2 | 0e1f853c | [summary](runs/2026-10-14_024441_mgo/summary.txt), [progress](runs/2026-10-14_024441_mgo/progress.txt), [changes](runs/2026-10-14_024441_mgo/changes.txt) |
| srtio3 | perovskite oxide, indirect R-Gamma gap from Ti 3d conduction bands | 2026-10-14T02:55 | pass | - | 1.839 (1.4-2.3) | 36 | 10.53 | 2a95e512 | 0e1f853c | [summary](runs/2026-10-14_024441_srtio3/summary.txt), [progress](runs/2026-10-14_024441_srtio3/progress.txt), [changes](runs/2026-10-14_024441_srtio3/changes.txt) |
| cu | d-band metal, needs smearing | 2026-10-14T02:57 | pass | - | -2.366 (metal) | 31 | 8.47 | 9957ab83 | 0e1f853c | [summary](runs/2026-10-14_024910_cu/summary.txt), [progress](runs/2026-10-14_024910_cu/progress.txt), [changes](runs/2026-10-14_024910_cu/changes.txt) |
| fe | ferromagnetic metal; must be spin-polarized (~2.2 muB/atom) | 2026-10-14T03:09 | pass | - | -1.868 (metal) | 47 | 20.25 | ee656768 | 0e1f853c | [summary](runs/2026-10-14_024927_fe/summary.txt), [progress](runs/2026-10-14_024927_fe/progress.txt), [changes](runs/2026-10-14_024927_fe/changes.txt) |
| al | free-electron-like sp metal, needs smearing | 2026-10-14T02:58 | pass | - | -13.061 (metal) | 31 | 9.03 | 026f9af5 | 0e1f853c | [summary](runs/2026-10-14_024933_al/summary.txt), [progress](runs/2026-10-14_024933_al/progress.txt), [changes](runs/2026-10-14_024933_al/changes.txt) |
| ni | weak itinerant ferromagnet (~0.6 muB/atom) | 2026-10-14T02:59 | **FAIL** | magnetization | -1.002 (metal) | 34 | 10.18 | 1095dd8f | 0e1f853c | [summary](runs/2026-10-14_024946_ni/summary.txt), [progress](runs/2026-10-14_024946_ni/progress.txt), [changes](runs/2026-10-14_024946_ni/changes.txt) |
| hbn | layered wide-gap insulator (same space group as graphite, not a metal) | 2026-10-14T03:02 | **FAIL** | band_gap | 3.712 (3.4-4.6) | 37 | 9.19 | 9d58e85a | 0e1f853c | [summary](runs/2026-10-14_025316_hbn/summary.txt), [progress](runs/2026-10-14_025316_hbn/progress.txt), [changes](runs/2026-10-14_025316_hbn/changes.txt) |
| lif | widest-gap ionic crystal; light atoms, Li 1s semicore | 2026-10-14T03:06 | pass | - | 8.589 (8.0-9.8) | 39 | 11.67 | a80c617d | 0e1f853c | [summary](runs/2026-10-14_025504_lif/summary.txt), [progress](runs/2026-10-14_025504_lif/progress.txt), [changes](runs/2026-10-14_025504_lif/changes.txt) |
| cdte | II-VI semiconductor with Cd 4d states; spin-orbit optional (Te Z=52) | 2026-10-14T03:05 | pass | - | 0.549 (0.1-1.1) | 34 | 10.0 | 81728f68 | 0e1f853c | [summary](runs/2026-10-14_025522_cdte/summary.txt), [progress](runs/2026-10-14_025522_cdte/progress.txt), [changes](runs/2026-10-14_025522_cdte/changes.txt) |
| pbte | heavy-element narrow gap at L; spin-orbit required | 2026-10-14T03:08 | pass | - | 0.054 (0.02-0.7) | 36 | 12.22 | 25fa0bbd | 0e1f853c | [summary](runs/2026-10-14_025632_pbte/summary.txt), [progress](runs/2026-10-14_025632_pbte/progress.txt), [changes](runs/2026-10-14_025632_pbte/changes.txt) |
| bi2se3 | layered topological insulator; spin-orbit required (inverts the gap at Gamma) | 2026-10-14T03:23 | pass | - | 0.123 (0.1-0.6) | 41 | 25.45 | 892ab881 | 0e1f853c | [summary](runs/2026-10-14_025748_bi2se3/summary.txt), [progress](runs/2026-10-14_025748_bi2se3/progress.txt), [changes](runs/2026-10-14_025748_bi2se3/changes.txt) |

## History

### zno: polar hexagonal cell, no inversion centre; Zn 3d semicore

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T02:49 | sweep 6: repeatability of all 14 + stretch suite (never tuned) | vllm | pass | - | 0.755 | 45 | 15.78 | 13.74 | 4e6cf1a5 | 0e1f853c |
| 2026-10-14T02:17 | sweep 5: magnetic moments on the cell + validator; golden references active | vllm | pass | - | 0.755 | 41 | 11.36 | 10.03 | 4e6cf1a5 | 61135fec |
| 2026-10-14T01:50 | sweep 4: QE pinned to dedicated GPUs 2-3 | vllm | pass | - | 0.755 | 35 | 9.42 | 8.3 | a34e4452 | 1b8c79f4 |
| 2026-10-14T00:53 | sweep 2: qe_output reader, tick lists, lsym off, 3 QE slots; extended baseline | vllm | pass | - | 0.754 | 39 | 17.42 | 5.52 | b86da611 | e3df222f |
| 2026-10-14T00:25 | sweep 1: vLLM + LangGraph, 1ebd4df plot fixes | vllm | pass | - | 0.754 | 47 | 38.65 | 6.23 | 5e9cf136 | aa252971 |
| 2026-10-07T00:26 | test34 rerun (in-process transformers, before the benchmark existed) | qwen | pass | - | 0.754 | 54 | 42.64 | - | - | 8030a6a9 |

### gaas: III-V with a small direct gap (PBE nearly closes it)

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T02:44 | sweep 6: repeatability of all 14 + stretch suite (never tuned) | vllm | pass | - | 0.075 | 34 | 10.48 | 9.37 | 77d6e749 | 0e1f853c |
| 2026-10-14T02:12 | sweep 5: magnetic moments on the cell + validator; golden references active | vllm | pass | - | 0.075 | 29 | 6.96 | 5.96 | 77d6e749 | 61135fec |
| 2026-10-14T01:48 | sweep 4: QE pinned to dedicated GPUs 2-3 | vllm | pass | - | 0.076 | 35 | 7.77 | 6.57 | bf159bbf | 1b8c79f4 |
| 2026-10-14T00:29 | sweep 1: vLLM + LangGraph, 1ebd4df plot fixes | vllm | pass | - | 0.077 | 48 | 42.94 | 8.92 | 1f0e056a | aa252971 |
| 2026-10-07T01:46 | test34 rerun (in-process transformers, before the benchmark existed) | qwen | pass | - | 0.077 | 55 | 80.21 | - | - | 8030a6a9 |

### graphite: semimetal, needs smearing; layered hexagonal

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T02:49 | sweep 6: repeatability of all 14 + stretch suite (never tuned) | vllm | pass | - | -0.005 | 61 | 15.69 | 13.28 | 843005b9 | 0e1f853c |
| 2026-10-14T02:13 | sweep 5: magnetic moments on the cell + validator; golden references active | vllm | pass | - | -0.005 | 32 | 7.68 | 6.52 | 843005b9 | 61135fec |
| 2026-10-14T01:47 | sweep 4: QE pinned to dedicated GPUs 2-3 | vllm | pass | - | -0.005 | 26 | 6.36 | 5.16 | a5f858fe | 1b8c79f4 |
| 2026-10-14T01:02 | sweep 2: qe_output reader, tick lists, lsym off, 3 QE slots; extended baseline | vllm | pass | - | -0.005 | 32 | 26.8 | 5.33 | 24c763aa | e3df222f |
| 2026-10-13T23:58 | sweep 1: vLLM + LangGraph, 1ebd4df plot fixes | vllm | pass | - | -0.005 | 50 | 12.33 | 7.06 | 21ba0266 | aa252971 |
| 2026-10-07T02:18 | test34 rerun (in-process transformers, before the benchmark existed) | qwen | pass | - | -0.005 | 40 | 31.94 | - | - | 8030a6a9 |

### cspbbr3: heavy elements, spin-orbit required (it shrinks the gap by ~1 eV)

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T02:49 | sweep 6: repeatability of all 14 + stretch suite (never tuned) | vllm | pass | - | 0.390 | 43 | 15.95 | 11.46 | 2d3731e5 | 0e1f853c |
| 2026-10-14T02:25 | sweep 5: magnetic moments on the cell + validator; golden references active | vllm | pass | - | 0.384 | 54 | 19.73 | 15.11 | 2d3731e5 | 61135fec |
| 2026-10-14T01:54 | sweep 4: QE pinned to dedicated GPUs 2-3 | vllm | pass | - | 0.390 | 36 | 13.34 | 8.93 | 1e677c8d | 1b8c79f4 |
| 2026-10-14T00:22 | sweep 1: vLLM + LangGraph, 1ebd4df plot fixes | vllm | pass | - | 0.373 | 46 | 36.3 | 6.62 | 5763242e | aa252971 |
| 2026-10-07T03:17 | test34 rerun (in-process transformers, before the benchmark existed) | qwen | pass | - | 0.386 | 58 | 59.15 | - | - | 8030a6a9 |

### mos2: layered dichalcogenide with an indirect gap

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T02:44 | sweep 6: repeatability of all 14 + stretch suite (never tuned) | vllm | pass | - | 1.203 | 42 | 10.87 | 8.94 | abdf34dc | 0e1f853c |
| 2026-10-14T02:15 | sweep 5: magnetic moments on the cell + validator; golden references active | vllm | pass | - | 1.203 | 36 | 9.72 | 7.61 | abdf34dc | 61135fec |
| 2026-10-14T01:52 | sweep 4: QE pinned to dedicated GPUs 2-3 | vllm | pass | - | 1.203 | 39 | 11.26 | 8.92 | e0c22990 | 1b8c79f4 |
| 2026-10-14T00:54 | sweep 2: qe_output reader, tick lists, lsym off, 3 QE slots; extended baseline | vllm | pass | - | 1.203 | 42 | 18.67 | 6.58 | 1bc2d5b7 | e3df222f |
| 2026-10-14T00:31 | sweep 1: vLLM + LangGraph, 1ebd4df plot fixes | vllm | pass | - | 1.203 | 63 | 45.29 | 6.84 | 499c9704 | aa252971 |
| 2026-10-07T03:55 | test34 rerun (in-process transformers, before the benchmark existed) | qwen | pass | - | 1.203 | 47 | 38.37 | - | - | 8030a6a9 |

### tio2: reference benchmark; direct gap at Gamma, checked at every high-symmetry point

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T02:44 | sweep 6: repeatability of all 14 + stretch suite (never tuned) | vllm | pass | - | 1.848 | 35 | 10.78 | 9.17 | 3861f7c7 | 0e1f853c |
| 2026-10-14T02:19 | sweep 5: magnetic moments on the cell + validator; golden references active | vllm | pass | - | 1.848 | 41 | 13.45 | 11.82 | 3861f7c7 | 61135fec |
| 2026-10-14T01:48 | sweep 4: QE pinned to dedicated GPUs 2-3 | vllm | pass | - | 1.848 | 32 | 8.06 | 6.13 | 386d0dec | 1b8c79f4 |
| 2026-10-14T00:45 | sweep 2: qe_output reader, tick lists, lsym off, 3 QE slots; extended baseline | vllm | pass | - | 1.730 | 33 | 9.66 | 4.45 | d780f938 | e3df222f |
| 2026-10-13T23:54 | sweep 1: vLLM + LangGraph, 1ebd4df plot fixes | vllm | pass | - | 1.838 | 30 | 8.32 | 4.57 | 6868c3d7 | aa252971 |

### si: textbook indirect gap (CBM near X, not at a labelled point)

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T02:44 | sweep 6: repeatability of all 14 + stretch suite (never tuned) | vllm | pass | - | 0.472 | 39 | 10.88 | 9.89 | 6e0d58cc | 0e1f853c |
| 2026-10-14T02:15 | sweep 5: magnetic moments on the cell + validator; golden references active | vllm | pass | - | 0.472 | 38 | 9.42 | 8.37 | 6e0d58cc | 61135fec |
| 2026-10-14T01:46 | sweep 4: QE pinned to dedicated GPUs 2-3 | vllm | pass | - | 0.472 | 31 | 5.93 | 5.13 | 116c1e3e | 1b8c79f4 |
| 2026-10-14T01:37 | sweep 3: dedicated QE GPUs, scf_kmesh, magnetism rule | vllm | pass | - | 0.473 | 33 | 22.13 | 7.41 | 7543c481 | 15c2b0a2 |
| 2026-10-14T00:44 | sweep 2: qe_output reader, tick lists, lsym off, 3 QE slots; extended baseline | vllm | pass | - | 0.472 | 34 | 8.74 | 4.54 | f8789fb5 | e3df222f |

### diamond: wide indirect gap; degenerate valence pairs at X

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T02:49 | sweep 6: repeatability of all 14 + stretch suite (never tuned) | vllm | pass | - | 4.055 | 41 | 15.41 | 14.05 | ff2577f2 | 0e1f853c |
| 2026-10-14T02:27 | sweep 5: magnetic moments on the cell + validator; golden references active | vllm | pass | - | 4.055 | 44 | 14.35 | 13.19 | ff2577f2 | 61135fec |
| 2026-10-14T01:55 | sweep 4: QE pinned to dedicated GPUs 2-3 | vllm | pass | - | 4.055 | 28 | 8.82 | 7.92 | c0591c8d | 1b8c79f4 |
| 2026-10-14T00:57 | sweep 2: qe_output reader, tick lists, lsym off, 3 QE slots; extended baseline | vllm | pass | - | 4.055 | 27 | 12.99 | 5.14 | e982a950 | e3df222f |

### nacl: ionic wide-gap insulator, alkali semicore

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T02:53 | sweep 6: repeatability of all 14 + stretch suite (never tuned) | vllm | pass | - | 5.039 | 34 | 8.55 | 7.61 | 28568d74 | 0e1f853c |
| 2026-10-14T02:26 | sweep 5: magnetic moments on the cell + validator; golden references active | vllm | pass | - | 5.039 | 38 | 10.59 | 9.62 | 28568d74 | 61135fec |
| 2026-10-14T01:59 | sweep 4: QE pinned to dedicated GPUs 2-3 | vllm | pass | - | 5.039 | 41 | 10.72 | 9.71 | a2a1bc08 | 1b8c79f4 |
| 2026-10-14T01:05 | sweep 2: qe_output reader, tick lists, lsym off, 3 QE slots; extended baseline | vllm | pass | - | 5.038 | 31 | 12.48 | 4.93 | 078b8223 | e3df222f |

### gan: nitride with Ga 3d semicore states; direct gap at Gamma

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T02:56 | sweep 6: repeatability of all 14 + stretch suite (never tuned) | vllm | pass | - | 1.877 | 42 | 12.13 | 10.59 | 498c9c97 | 0e1f853c |
| 2026-10-14T02:24 | sweep 5: magnetic moments on the cell + validator; golden references active | vllm | pass | - | 1.877 | 33 | 10.24 | 8.87 | 498c9c97 | 61135fec |
| 2026-10-14T02:01 | sweep 4: QE pinned to dedicated GPUs 2-3 | vllm | pass | - | 1.877 | 45 | 14.11 | 12.77 | 0eda10c0 | 1b8c79f4 |
| 2026-10-14T01:07 | sweep 2: qe_output reader, tick lists, lsym off, 3 QE slots; extended baseline | vllm | pass | - | 1.866 | 36 | 21.4 | 5.31 | d86c1598 | e3df222f |

### mgo: ionic oxide, direct gap at Gamma

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T02:54 | sweep 6: repeatability of all 14 + stretch suite (never tuned) | vllm | pass | - | 4.782 | 37 | 10.24 | 9.35 | af9867f2 | 0e1f853c |
| 2026-10-14T02:25 | sweep 5: magnetic moments on the cell + validator; golden references active | vllm | pass | - | 4.782 | 33 | 9.95 | 9.0 | af9867f2 | 61135fec |
| 2026-10-14T01:58 | sweep 4: QE pinned to dedicated GPUs 2-3 | vllm | pass | - | 4.782 | 38 | 9.42 | 8.59 | ecccdae1 | 1b8c79f4 |

### cu: d-band metal, needs smearing

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T02:57 | sweep 6: repeatability of all 14 + stretch suite (never tuned) | vllm | pass | - | -2.366 | 31 | 8.47 | 7.37 | 9957ab83 | 0e1f853c |
| 2026-10-14T02:28 | sweep 5: magnetic moments on the cell + validator; golden references active | vllm | pass | - | -2.366 | 27 | 9.37 | 8.31 | 9957ab83 | 61135fec |
| 2026-10-14T02:00 | sweep 4: QE pinned to dedicated GPUs 2-3 | vllm | pass | - | -2.366 | 29 | 8.24 | 7.07 | ca08fb33 | 1b8c79f4 |

### srtio3: perovskite oxide, indirect R-Gamma gap from Ti 3d conduction bands

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T02:55 | sweep 6: repeatability of all 14 + stretch suite (never tuned) | vllm | pass | - | 1.839 | 36 | 10.53 | 9.11 | 2a95e512 | 0e1f853c |
| 2026-10-14T02:28 | sweep 5: magnetic moments on the cell + validator; golden references active | vllm | pass | - | 1.839 | 41 | 10.69 | 9.34 | 2a95e512 | 61135fec |
| 2026-10-14T02:02 | sweep 4: QE pinned to dedicated GPUs 2-3 | vllm | pass | - | 1.839 | 41 | 11.9 | 10.53 | 7474fe80 | 1b8c79f4 |

### fe: ferromagnetic metal; must be spin-polarized (~2.2 muB/atom)

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T03:09 | sweep 6: repeatability of all 14 + stretch suite (never tuned) | vllm | pass | - | -1.868 | 47 | 20.25 | 17.83 | ee656768 | 0e1f853c |
| 2026-10-14T02:31 | sweep 5: magnetic moments on the cell + validator; golden references active | vllm | pass | - | -1.868 | 31 | 7.51 | 5.78 | ee656768 | 61135fec |
| 2026-10-14T02:03 | sweep 4: QE pinned to dedicated GPUs 2-3 | vllm | **FAIL** | magnetization | -1.883 | 30 | 8.74 | 6.86 | 14ff4b5c | 1b8c79f4 |

### al: free-electron-like sp metal, needs smearing

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T02:58 | sweep 6: repeatability of all 14 + stretch suite (never tuned) | vllm | pass | - | -13.061 | 31 | 9.03 | 7.99 | 026f9af5 | 0e1f853c |

### ni: weak itinerant ferromagnet (~0.6 muB/atom)

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T02:59 | sweep 6: repeatability of all 14 + stretch suite (never tuned) | vllm | **FAIL** | magnetization | -1.002 | 34 | 10.18 | 9.1 | 1095dd8f | 0e1f853c |

### hbn: layered wide-gap insulator (same space group as graphite, not a metal)

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T03:02 | sweep 6: repeatability of all 14 + stretch suite (never tuned) | vllm | **FAIL** | band_gap | 3.712 | 37 | 9.19 | 7.99 | 9d58e85a | 0e1f853c |

### cdte: II-VI semiconductor with Cd 4d states; spin-orbit optional (Te Z=52)

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T03:05 | sweep 6: repeatability of all 14 + stretch suite (never tuned) | vllm | pass | - | 0.549 | 34 | 10.0 | 8.69 | 81728f68 | 0e1f853c |

### lif: widest-gap ionic crystal; light atoms, Li 1s semicore

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T03:06 | sweep 6: repeatability of all 14 + stretch suite (never tuned) | vllm | pass | - | 8.589 | 39 | 11.67 | 10.56 | a80c617d | 0e1f853c |

### pbte: heavy-element narrow gap at L; spin-orbit required

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T03:08 | sweep 6: repeatability of all 14 + stretch suite (never tuned) | vllm | pass | - | 0.054 | 36 | 12.22 | 10.22 | 25fa0bbd | 0e1f853c |

### bi2se3: layered topological insulator; spin-orbit required (inverts the gap at Gamma)

| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-14T03:23 | sweep 6: repeatability of all 14 + stretch suite (never tuned) | vllm | pass | - | 0.123 | 41 | 25.45 | 9.95 | 892ab881 | 0e1f853c |
