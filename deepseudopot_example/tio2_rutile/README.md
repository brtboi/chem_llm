# Rutile TiO2: DFT output for the DeePseudopot agent

Input for `chem_llm.dpp.agent` (`chem-llm-dpp`): a finished chem_llm band-structure run, and
a working `NN_config.par` to start from.

```bash
chem-llm-dpp deepseudopot_example/tio2_rutile/qe_run --work-dir runs/tio2_dpp --backend vllm
```

## qe_run/

A copy of `bench/work/tio2/calculations/000` from benchmark sweep 8 (run
`20261014-041640-7a196f`, 2026-10-14). It passed every benchmark check: every high-symmetry
point is within 0.15 eV of the golden reference, and the gap is 1.847 eV.

- rutile TiO2, mp-2657, primitive cell: 2 Ti + 4 O.
- PBEsol, Pseudo-Dojo norm-conserving scalar-relativistic pseudopotentials (`Ti.upf`, `O.upf`).
  Ti carries its 3s/3p semicore shell (z_valence 12), so the 6-atom cell has 48 electrons in
  24 occupied bands.
- No spin-orbit; fixed occupations.
- Bands: 48 bands on the seekpath path Γ–X–M–Γ–Z–R–A–Z|X–R|M–A (304 points, `kpath.json`).

The wavefunctions (`tmp/`) are left out: nothing downstream needs them.

**Band groups** (from `000.bands.dat`):

| DFT bands | states | energies (eV) |
|---|---|---|
| 0–1 | Ti 3s | −47.6 |
| 2–7 | Ti 3p | −24.3 to −23.9 |
| 8–11 | O 2s | −9.1 to −7.7 |
| 12–23 | O 2p, the valence band maximum at 9.00 | up to 9.00 |
| 24– | Ti 3d conduction bands, minimum at 10.85 | 10.85 and up |

An empirical pseudopotential describes the O 2s/2p valence and the Ti 3d conduction bands. It
does not describe the Ti 3s/3p semicore. DeePseudopot fits its lowest eigenvalues to the
reference columns in order, so the fit has to start at DFT band 8 (system prompt, INVARIANT 2).
The earlier one-shot attempt in `sandbox/test_claude` fed it all 48 bands from band 0.

## NN_config_template.par

The `NN_config.par` the agent starts from: single system, no spin-orbit, `num_cores = 0`
(>= 1 crashes every epoch and runs out of memory), `init_Zunger_num_epochs = 2000` (the
gradient stage first fits the network to the Zunger form it starts from), `optimizer_lr = 0.01`.

## Training: a physical start, Monte Carlo, then gradient descent

Measured by hand on this run (bands 8–31, the six unique high-symmetry points, `maxKE = 2.9` Ha,
about 100 plane waves; gap bands weighted 4x). RMS error over all fitted bands, against DFT:

| stage | time | RMS error | gap (DFT 1.85 eV) |
|---|---|---|---|
| all-zero start, 600 gradient epochs (the agent's first run) | 9 min | 1.59 eV | −2.64 eV |
| random network, 40 Monte Carlo steps on its weights | 1 min | ~14 eV, loss down 3% | — |
| start from the UPF files (`initial_zunger_params`), no training | — | 16.7 eV | — |
| + Monte Carlo over the Zunger parameters, 300 + 3000 steps | 4.5 min | 1.84 eV | −1.84 eV |
| + gradient descent from that, 600 epochs | 10 min | 1.08 eV (flat after 400) | −1.33 eV |

- **Start:** each element's local potential from its UPF file, with Ti's 3s/3p shells counted as
  core (Ti becomes a +4 ion, O is +6), Lindhard-screened by the 32 electrons in the fitted bands.
  As screened local potentials they bind 8 spurious Ti s/p-like states below O 2s: 12 deep bands
  where DFT has 4.
- **Monte Carlo over the 4 Zunger parameters per element** (`main_mc_Zunger.py`, about 0.07 s
  per step) fixes the band groups: 4 O 2s, 12 O 2p, then the conduction bands. Monte Carlo over
  the network's weights (`mc_bool = 1`) is about 1.5 s per step and barely moves from a random
  network.
- **The gap stays inverted.** Evaluating the Monte Carlo potential at `maxKE` 2.9, 4.5 and 6 Ha
  gives −1.84, −1.67 and −1.65 eV, so the basis is not the limit. A purely local potential does not
  place the Ti 3d conduction bands; the non-local terms of the DeePseudopot checkout chem_llm
  vendors (parameters 6–7, l = 1, added only in the spin-orbit Hamiltonian) cannot either.
