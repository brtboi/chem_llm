#!/usr/bin/env python3
"""Entry point shim -- kept at the repo root so `python replay_copy.py`
works (see main.py for why this shim pattern exists). All real logic lives
in chem_llm/replay.py.

Replays a recorded run from one sandbox/testN directory into another,
identified by their integer suffixes, creating the destination if needed.

CLI usage:
    python replay_copy.py 17 25                # replay test17's latest run into sandbox/test25
    python replay_copy.py 17 25 --index 0       # replay test17's first recorded run
    python replay_copy.py 17 25 --clear-dir     # clear sandbox/test25 first
"""
from chem_llm.replay import copy_main

if __name__ == "__main__":
    copy_main()
