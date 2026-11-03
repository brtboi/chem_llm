# Rutile TiO2: DFT output for the DeePseudopot agent

Input for `chem_llm.dpp_agent` (`chem-llm-dpp`): a finished chem_llm band-structure run, and
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
(>= 1 crashes every epoch and runs out of memory). It also sets
`init_Zunger_num_epochs = 0`, because no fitted Zunger parameters exist for Ti or O, so
training starts from zero.

Measured by hand on this run before the agent's test: a bundle of bands 8–33 at the six
unique high-symmetry points, `maxKE = 4` Ha, starting from zero. Training takes about 1 s per
epoch on the CPU:

| `optimizer_lr` | epochs | RMS error | gap (fit vs DFT 1.85 eV) |
|---|---|---|---|
| 0.001 | 4 | 15.2 eV | −5.7 eV |
| 0.01 | 100 | 7.2 eV | −0.9 eV |
| 0.01 | 200 | 3.8 eV | −2.3 eV |

So 0.01 is the template's rate. From zero, a usable fit needs many hundreds of epochs, or a
better starting point than zero.
