"""Run the real agent loop without a GPU, the network or Quantum ESPRESSO.

    from chem_llm import DFTAgent
    from chem_llm.testing import ScriptedChatModel, mock_tools, fake_qe_workflow

    llm = ScriptedChatModel([
        {"tool": "write_file", "args": {"path": "hello.py", "content": "print('hi')"}},
        {"tool": "run_python", "args": {"path": "hello.py"}},
        {"tool": "done", "args": {"summary": "ok"}},
    ])
    with mock_tools(run_espresso_workflow=fake_qe_workflow()):
        result = DFTAgent(llm=llm).run("Si", "Fd-3m", work_dir="/tmp/x")

Everything else is the production path: the LangGraph loop, harness
refusals, history trimming, prompt building, telemetry, the log record,
trace.jsonl and the run summary. `llm.seen` holds the messages the model
was sent at each step, so a test can assert on the prompt itself.
"""
from __future__ import annotations

import contextlib
import json
from typing import Any, Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field


class ScriptedChatModel(BaseChatModel):
    """Replies with a fixed list of tool calls (dicts, sent as JSON) or raw
    strings (to test malformed output), one per call. An Exception in the
    list is raised instead of replying. Raises RuntimeError when the script
    runs out. Reports token usage as characters / 4."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    replies: list[Any]
    model_name: str = "scripted"
    seen: list[list] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        if len(self.seen) > len(self.replies):
            raise RuntimeError(f"ScriptedChatModel: script exhausted after {len(self.replies)} replies")
        reply = self.replies[len(self.seen) - 1]
        if isinstance(reply, Exception):
            raise reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        n_in = sum(len(m.content) for m in messages) // 4
        n_out = max(1, len(text) // 4)
        usage = {"input_tokens": n_in, "output_tokens": n_out, "total_tokens": n_in + n_out}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text, usage_metadata=usage))])


@contextlib.contextmanager
def mock_tools(**replacements) -> Iterator[None]:
    """Swap tool implementations for the duration of the block, e.g.
    mock_tools(run_espresso_workflow=fake_qe_workflow(), generate_cif=...).
    The tool schemas the model sees are unchanged."""
    from .tools import TOOL_DISPATCH

    unknown = set(replacements) - set(TOOL_DISPATCH)
    if unknown:
        raise KeyError(f"no such tool(s): {sorted(unknown)}")
    saved = {name: TOOL_DISPATCH[name] for name in replacements}
    TOOL_DISPATCH.update(replacements)
    try:
        yield
    finally:
        TOOL_DISPATCH.update(saved)


def fake_qe_workflow(success: bool = True, timings_s: dict | None = None):
    """A stand-in for run_espresso_workflow that returns at once with the
    same result shape (and per-stage timings) as the real tool."""
    timings = timings_s or {"scf": 1.0, "bands_scf": 2.0, "bands_post": 0.5}

    def run_espresso_workflow(directory: str, pw_input: str = "pw.in", bands_input: str = "bands.in",
                              bands_post_input: str = "bands_post.in"):
        if not success:
            return {"success": False, "stderr": "scf (pw.x -in pw.in) exited with code 1",
                    "completed_steps": [], "timings_s": {"scf": timings.get("scf", 0.0)}}
        return {"success": True, "directory": directory, "completed_steps": list(timings),
                "outputs": {}, "timings_s": dict(timings)}

    return run_espresso_workflow
