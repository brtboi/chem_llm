"""Against a real vLLM server (scripts/serve_vllm.sh). Skipped when none is up.

    pytest -m vllm
"""
import os

import pytest

from chem_llm.llm import check_server, vllm_chat_model

URL = os.environ.get("CHEM_LLM_VLLM_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("CHEM_LLM_VLLM_MODEL", "Qwen/Qwen3.8-27B")

try:
    SERVED = check_server(URL, timeout=2)
except ConnectionError:
    SERVED = []

pytestmark = [pytest.mark.vllm, pytest.mark.skipif(MODEL not in SERVED, reason=f"no vLLM server serving {MODEL} at {URL}")]


def test_one_agent_step_is_a_valid_tool_call():
    from chem_llm.agent_core import make_chat_step_source
    from chem_llm.state import AgentState
    from chem_llm.tasks import build_task

    state = AgentState(build_task(compound="Si", spacegroup="Fd-3m"))
    llm = vllm_chat_model(URL, MODEL, max_tokens=2000, temperature=0.3, do_sample=True)
    call = make_chat_step_source(llm, verbose=False)(state, 1)
    assert call is not None and "tool" in call, state.history
    assert state.llm_usage["prompt_tokens"] > 5000 and state.llm_usage["completion_tokens"] > 0
