#!/usr/bin/env python3
"""Entry point shim -- kept at the repo root so `python main.py` still
works. All real logic lives in chem_llm/main.py; this just delegates to
the installed package (see pyproject.toml's [build-system] -- `uv sync`
installs chem_llm in editable mode, so `import chem_llm` resolves from
any working directory, not just when this file happens to be script[0]).
"""
from chem_llm.main import main

if __name__ == "__main__":
    main()
