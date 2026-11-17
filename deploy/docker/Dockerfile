# The chem-llm DFT agent with GPU Quantum ESPRESSO. The model is served by a
# separate vLLM container (compose.yaml); this image only talks to it over HTTP.
#
#   docker build -f deploy/docker/Dockerfile -t chem-llm-agent .     (from the repo root)
#
# Base: NVIDIA's NGC Quantum ESPRESSO image (GPU build of pw.x / bands.x, OpenMPI).
# qe-7.3.1 is the newest NGC tag (Perlmutter runs 7.5); needs NVIDIA driver >= 550,
# Pascal through Hopper GPUs. Check https://catalog.ngc.nvidia.com/orgs/hpc/containers/quantum_espresso
FROM nvcr.io/hpc/quantum_espresso:qe-7.3.1

# uv installs its own Python 3.12 (chem-llm needs >= 3.12, whatever the base image
# ships) and the locked dependencies from uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.11.25 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /opt/chem_llm
# dependencies first, so code edits don't reinstall torch
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project --python 3.12
COPY chem_llm ./chem_llm
RUN uv sync --frozen --no-dev --python 3.12

ENV PATH=/opt/chem_llm/.venv/bin:$PATH \
    HF_HOME=/data/hf \
    MPLBACKEND=Agg \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# The container's config (QE paths, vLLM endpoint, /data paths). Runs are found
# by Settings.discover() from the working directory /work.
COPY deploy/docker/chem_llm.yaml /work/chem_llm.yaml
WORKDIR /work

# QE binaries must be on PATH (the NGC image puts them there); fail the build if not.
RUN command -v pw.x && command -v bands.x

ENTRYPOINT ["chem-llm"]
CMD ["--help"]
