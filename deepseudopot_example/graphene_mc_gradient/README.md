# Graphene: Monte Carlo, then gradient descent (reference configurations)

Configurations from two of a collaborator's DeePseudopot runs on graphene (June 2026,
`monte_carlo_2/` and `nn_grad_attr/`), kept as a reference for the two training modes of
DeePseudopot's `main.py`. They are not a worked example of the two stages succeeding one after
the other; see "What the runs showed".

## mc/: Monte Carlo on the network weights (`mc_bool = 1`)

The input bundle of `monte_carlo_2/inputs_mc_small` (graphene, 2 C atoms, 8 bands, 23 k-points,
`maxKE = 15` Ha, non-local l=1 terms on).

| key | value | meaning |
|---|---|---|
| `mc_bool` | 1 | run `runMC_NN` instead of gradient training (`max_num_epochs` is ignored) |
| `mc_iter` | 1000 | Monte Carlo steps |
| `mc_percentage` | 0.05 | each step multiplies every network weight by a random factor in [0.95, 1.05] (and the SOC/NL prefactors by [1 - p/100, 1 + p/100]) |
| `mc_beta` | 5 | Metropolis: a worse step is accepted with probability exp(-beta (sqrt(new) - sqrt(current))) |

A step that lowers the loss is always kept; `best_PPmodel.pth` is the lowest-loss model and
`final_mc_cost.dat` logs every step (`iter newLoss accept? bestLoss currLoss`). The run started
from an `init_PPmodel.pth` and `init_qSpace_pot.par` (a tabulated V(q), the format in
`../tio2_rutile/`), not included here.

## gradient/: gradient descent (`mc_bool = 0`)

`nn_grad_attr/inputs_nonloc_small_26`: the same system at `maxKE = 16`, `optimizer_lr = 1e-4`,
`max_num_epochs = 250`, a 20-20-20 network, starting from an `init_PPmodel.pth`.
`final_training_cost.dat` is its cost per epoch.

Three of its keys are **not in the DeePseudopot checkout chem_llm vendors** (`main`): `partial_eig`,
`NLbool`, `nonlocal_grad` / `nonlocal_grad_lr` (gradient training of the non-local prefactors)
exist only on the upstream `irrep-block` and `spin-polarized` branches. Our `main` reads the file
and ignores them. These runs also used `num_cores = 32`; on our `main`, `num_cores >= 1` with
`local_env_corr = 0` crashes every epoch, so `write_bundle` refuses it.

## Passing a model from one stage to the next

`main.py` runs one mode per call. To continue from a model, copy it into the next stage's input
folder as `init_PPmodel.pth`: `init_ZungerPP` loads it and, with `force_retrain = 0`, skips the
pre-fit. The network shape (`hiddenLayers`, `PPmodel`) must match the saved model's, or the load
fails.

## What the runs showed

- Monte Carlo: 521 of 1000 steps before the 4-hour job limit (about 27 s per step on 23 CPU
  cores); best loss 747 -> 653, 113 of 521 steps accepted.
- Gradient descent: training cost 900 -> 874 over 209 epochs. It did not start from the Monte
  Carlo model (different network shape), so the two losses are not one curve.

For TiO2, the chem_llm DeePseudopot agent runs the Monte Carlo stage over the Zunger-form
parameters (`main_mc_Zunger.py`) rather than the network weights; see `../tio2_rutile/README.md`.
