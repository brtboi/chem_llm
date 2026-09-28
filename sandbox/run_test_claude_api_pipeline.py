"""Rutile TiO2 from-scratch benchmark -- Claude API.

Paired with run_test27_pipeline.py (local Qwen). Everything shared (task
text, tool set, step budget, doc index) lives in tio2_task.py; the only
difference is that the tool calls come from the Messages API instead of a
local model. Same system prompt, same JSON tool-call protocol, same loop.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tio2_task import DOC_INDEX_DIR, MAX_STEPS, TASK, hide_deepseudopot

os.environ["MATAGENT_WORK_DIR"] = "sandbox/test_claude_api"
os.environ["DOC_INDEX_DIR"] = DOC_INDEX_DIR

import json
from chem_llm import config
from dotenv import load_dotenv
load_dotenv()

hide_deepseudopot()

from chem_llm.agent_core import run_agent
from chem_llm.claude_backend import ClaudeModel, make_claude_step_source

if not config.ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is not set.")

# Model pinned to Opus 4.8 rather than the config default of Opus 5.
# Opus 5's safety classifier declines this harness's context with
# `reasoning_extraction`: rule 9 has the agent write its scientific
# reasoning into `note` steps, and replaying those notes back in the
# context reads as an attempt to extract reasoning traces. Measured on one
# mid-run context, Opus 5 refused 2 of 3 attempts (a first attempt at this
# benchmark died after 24 consecutive refusals) while Opus 4.8 and
# Sonnet 5 refused 0 of 3; server-side fallbacks did not rescue it.
# Opus 4.8 is the same tier and price as Opus 5.
#
# Effort is pinned explicitly rather than read from the environment:
# Claude Code sets CLAUDE_EFFORT in its own shell, which a run launched
# from here would otherwise silently inherit.
claude = ClaudeModel(model="claude-opus-4-8", effort="high", use_fallbacks=False)
print(f"Claude model: {claude.model} | effort: {claude.effort} | max_tokens: {claude.max_tokens}")

os.makedirs(config.WORK_DIR, exist_ok=True)
os.chdir(config.WORK_DIR)
print(f"Working directory: {os.getcwd()}")

final_state = run_agent(
    TASK, verbose=True,
    step_source=make_claude_step_source(claude, verbose=True),
    model_name=claude.model,
    clear_dir=True, load_path=None,  # no example directory
    max_steps=MAX_STEPS,
)

print("\nFINAL STATE:\n", json.dumps(final_state.to_dict(), indent=2))
