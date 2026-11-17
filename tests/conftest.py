import json
import os
from pathlib import Path

import pytest

# The tests need no network: keep scripted runs out of LangSmith even when .env turns
# tracing on (chem_llm.settings loads .env on import, but never overrides a set variable).
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from chem_llm import DFTAgent, Settings  # noqa: E402
from chem_llm.testing import ScriptedChatModel, fake_qe_workflow, mock_tools


@pytest.fixture
def settings(tmp_path):
    return Settings(work_root=tmp_path / "runs", doc_index_dir=tmp_path / "no_index")


@pytest.fixture
def run_scripted(settings, tmp_path):
    """run_scripted(replies, **run_kwargs) -> (RunResult, ScriptedChatModel):
    a full DFTAgent run driven by a scripted model, QE mocked."""

    def run(replies, work_dir=None, max_steps=None, **kwargs):
        llm = ScriptedChatModel(replies=replies)
        agent = DFTAgent(settings, llm=llm, max_steps=max_steps)
        with mock_tools(run_espresso_workflow=fake_qe_workflow()):
            result = agent.run("Si", "Fd-3m", work_dir=work_dir or tmp_path / "run", verbose=False, **kwargs)
        return result, llm

    return run


def records(work_dir: Path, name: str = "log.jsonl") -> list[dict]:
    return [json.loads(line) for line in (Path(work_dir) / name).read_text().splitlines() if line.strip()]


HELLO = [
    {"tool": "note", "args": {"text": "Plan: write and run a script."}},
    {"tool": "write_file", "args": {"path": "hello.py", "content": "print('hi')\n"}},
    {"tool": "run_python", "args": {"path": "hello.py"}},
    {"tool": "done", "args": {"summary": "printed hi"}},
]
