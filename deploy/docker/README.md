# Running the chem-llm DFT agent in Docker (e.g. on Google Cloud)

Two containers on one GPU machine, mirroring the Perlmutter setup (vLLM on its own GPU, QE on
another):

| container | image | GPU | what it does |
|---|---|---|---|
| `vllm` | `vllm/vllm-openai:v0.31.0` (stock) | 0 | serves Qwen3.8-27B over an OpenAI-compatible API |
| `agent` | `chem-llm-agent`, built from `Dockerfile` here | 1 | the chem-llm agent loop + GPU Quantum ESPRESSO |

Why two: vLLM pins its own torch (on Perlmutter it has its own venv for the same reason), and the
model server can stay up while agent containers come and go. Only the DFT agent is packaged;
DeePseudopot is excluded (`.dockerignore`).

Files: `Dockerfile` (agent image), `compose.yaml` (both services), `chem_llm.yaml` (the agent's
settings inside the container), and `/.dockerignore` at the repo root.

**Not yet tested.** No Docker on Perlmutter's login/compute nodes, so these files have not been
built or run. The first build on GCP is the test; check the items under "Verify on first build".

## What you need

### On Google Cloud
1. **A VM with 2 GPUs**, e.g. `a2-ultragpu-2g` (2x A100 80 GB).
   - Qwen3.8-27B in bf16 is ~54 GB of weights, so the model needs an 80 GB GPU to itself.
   - QE gets the second GPU. One GPU only: serve the model elsewhere, or run QE on CPU (see below).
2. **NVIDIA driver >= 550 and the NVIDIA Container Toolkit** on the host (the NGC QE image needs
   driver >= 550). A Deep Learning VM image has both; on plain Ubuntu, install them.
3. **A persistent disk of ~150 GB** for the `data` volume: model weights (~55 GB), the
   documentation index, and runs. Without it, every new VM re-downloads the model.
4. **Secrets:** `MP_API_KEY` (Materials Project) and `HF_TOKEN` (Hugging Face), from Secret
   Manager or the VM's environment. Never put them in the image or in `chem_llm.yaml`.
5. **Outbound internet** at run time: Materials Project lookups, pseudopotential downloads
   (pseudohub), and the first model/embedding-model download.
6. Optional: **Artifact Registry** to push the built `chem-llm-agent` image if more than one VM
   will use it.

### In the agent image (what `Dockerfile` does)
- Base `nvcr.io/hpc/quantum_espresso:qe-7.3.1`: GPU `pw.x`/`bands.x` with OpenMPI, Ampere/Hopper
  ready. That is the newest NGC tag; Perlmutter runs QE 7.5. Building GPU QE yourself needs the
  NVIDIA HPC SDK, which is the main reason to use this base.
- `uv` installs Python 3.12 and the exact dependencies of `uv.lock` (`--no-dev`).
- `chem_llm.yaml` copied to `/work`, the working directory, where `Settings.discover()` finds it.
- OpenMPI is allowed to run as root (`OMPI_ALLOW_RUN_AS_ROOT`), as containers usually do.

### In `chem_llm.yaml`
- QE: binaries on PATH, no Lmod `module`, no `srun` (`mpi_prefix: ""`, one GPU), `gpu_ids: [0]`
  (the one GPU the container sees).
- Model: `backend: vllm`, `vllm_url: http://vllm:8000/v1` (the compose service name).
- Paths: everything under the `/data` volume.

## Steps

```bash
# 0. on your machine: the code must be pushed (a clone gets origin, not local commits)
git clone git@github.com:brtboi/chem_llm.git && cd chem_llm

# 1. build the agent image (context = repo root)
docker compose -f deploy/docker/compose.yaml build agent

# 2. start the model server; the first start downloads ~55 GB, then loads for several minutes
export MP_API_KEY=... HF_TOKEN=...
docker compose -f deploy/docker/compose.yaml up -d vllm
docker compose -f deploy/docker/compose.yaml logs -f vllm      # wait for "Application startup complete"

# 3. build the documentation index once (scrapes the ASE/QE docs, downloads the embedding model)
docker compose -f deploy/docker/compose.yaml run --rm agent --build-docs

# 4. a run; results land in the data volume under /data/runs/TiO2
docker compose -f deploy/docker/compose.yaml run --rm agent TiO2 rutile --pretty-progress
```

Check a run's physics as on Perlmutter (`CLAUDE.md`): rutile TiO2 should give a gap of about
1.85 eV with the high-symmetry points matching `bench/references/tio2.json`.

## Verify on first build

- **The NGC tag and PATH:** the build fails at `command -v pw.x` if the binaries are not on PATH.
  NGC's page shows their helper script `run_qe.sh`; if `pw.x` is elsewhere, set `pw_x`/`bands_x`
  in `chem_llm.yaml` to absolute paths.
- **The vLLM image flags:** `--max-model-len 65536` (Perlmutter uses 131072 across 2 GPUs). On one
  80 GB GPU, lower it if vLLM reports too little KV-cache memory; agent prompts are ~20k tokens.
  Check the image accepts `--model` (older images use the OpenAI server entrypoint, newer
  `vllm serve`; both take it).
- **QE speed:** `omp_num_threads: 8` and a single GPU are a starting point. For several GPUs per
  QE job, use `mpi_prefix: "mpirun -np <n>"`, `pw_args: "-nk <n>"`.
- **Driver vs CUDA:** vLLM 0.31 on Perlmutter runs torch 2.13+cu130; the host driver must support
  the CUDA versions of both images.

## Variations

- **One GPU:** run QE on CPU. Replace the base image with e.g. `ubuntu:24.04` plus
  `apt-get install quantum-espresso openmpi-bin`, set `gpu_ids: null`,
  `mpi_prefix: "mpirun -np <cores>"`, and remove the agent's GPU reservation. CPU QE speed on
  these cells is unmeasured; time TiO2 first.
- **No Docker:** the same works on a VM with `git clone`, `uv sync`, a QE install, and vLLM in its
  own venv (`model.vllm_executable`, or `chem-llm --start-vllm`, which picks GPUs not in
  `gpu_ids`).
- **Claude instead of Qwen:** `backend: claude` with `ANTHROPIC_API_KEY` needs no model server
  or model GPU, but bills per step.
