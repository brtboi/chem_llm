#!/usr/bin/env bash
# Serve the agent's model with vLLM, as an OpenAI-compatible API that the
# "vllm" backend (model.backend: vllm in config.yaml) talks to.
#
#     scripts/serve_vllm.sh                 # foreground; Ctrl-C to stop
#     scripts/serve_vllm.sh > vllm.log 2>&1 &
#
# vLLM pins its own torch, so it lives in its own venv, not the package's:
#     uv venv --python 3.12 --managed-python $VLLM_VENV
#     VIRTUAL_ENV=$VLLM_VENV uv pip install vllm
# (on NERSC set UV_CACHE_DIR / UV_PYTHON_INSTALL_DIR to scratch: the home
# filesystem refuses uv's file locks with error 524.)
#
# Compared with in-process transformers, vLLM batches, uses paged attention
# and caches the prompt prefix -- every agent step re-sends the same ~8K-token
# system prompt, which prefix caching computes once. Several runs can share
# one server.
#
# Every setting below can be overridden from the environment.
set -euo pipefail

VLLM_VENV=${VLLM_VENV:-$PSCRATCH/venvs/vllm}
MODEL=${MODEL:-Qwen/Qwen3.8-27B}
PORT=${PORT:-8000}
# One shard per GPU on the node.
TP=${TP:-$(nvidia-smi -L | wc -l)}
# vLLM preallocates this fraction of every GPU (weights + KV cache). Quantum
# ESPRESSO runs on the same GPUs while the server is up, so leave it room:
# 0.75 of a 40 GB A100 is 30 GB for vLLM, ~10 GB for pw.x.
GPU_MEM=${GPU_MEM:-0.75}
# Longest prompt + reply. Agent prompts have reached ~71K tokens.
MAX_LEN=${MAX_LEN:-131072}

export HF_HOME=${HF_HOME:-$PSCRATCH/huggingface}
# vLLM, torch.compile and Triton cache compiled kernels under ~/.cache with
# file locks, which the NERSC home filesystem refuses (OSError 524).
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$PSCRATCH/.cache}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$XDG_CACHE_HOME/vllm}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$PSCRATCH}   # -> $PSCRATCH/.cache/flashinfer
# Weights are already cached; do not phone home on every start.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

# --limit-mm-per-prompt: the checkpoint is multimodal, the agent sends text
# only; zero images/videos skips the vision encoder's memory reservation.
exec "$VLLM_VENV/bin/vllm" serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len "$MAX_LEN" \
    --enable-prefix-caching \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    "$@"
