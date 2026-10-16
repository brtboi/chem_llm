import functools

import pytest

import chem_llm.main as cli
from chem_llm import DFTAgent
from chem_llm.replay import replay_run
from chem_llm.testing import ScriptedChatModel, fake_qe_workflow, mock_tools
from conftest import HELLO, records


@pytest.fixture
def scripted_cli(settings, tmp_path, monkeypatch, capsys):
    """Run `chem-llm ...` with the scripted model in place of the backend."""

    def run(*flags):
        agent = functools.partial(DFTAgent, llm=ScriptedChatModel(replies=HELLO))
        monkeypatch.setattr(cli, "DFTAgent", agent)
        monkeypatch.setattr(cli.Settings, "discover", classmethod(lambda cls: settings))
        with mock_tools(run_espresso_workflow=fake_qe_workflow()):
            code = cli.main(["Si", "Fd-3m", "--work-dir", str(tmp_path / "cli"), *flags])
        out, err = capsys.readouterr()
        return code, out, err

    return run


def test_pretty_progress_streams_readable_lines(scripted_cli):
    code, out, _ = scripted_cli("--pretty-progress")
    assert "running hello.py ..." in out and "RAW MODEL OUTPUT" not in out


def test_summarize_keeps_stdout_for_the_summary(scripted_cli):
    code, out, err = scripted_cli("--summarize")
    assert out.startswith("chem_llm run summary") and "5. Where the time went" in out
    assert "RAW MODEL OUTPUT" in err


def test_pretty_progress_with_summarize(scripted_cli):
    code, out, err = scripted_cli("--pretty-progress", "--summarize")
    assert out.startswith("chem_llm run summary") and "running hello.py ..." in err


def test_replay_reexecutes_recorded_tools(run_scripted, tmp_path):
    run_scripted(HELLO)
    entry = records(tmp_path / "run")[-1]
    with mock_tools(run_espresso_workflow=fake_qe_workflow()):
        state = replay_run(entry, tmp_path / "replayed", verbose=False)
    assert [e["tool"] for e in state.history] == ["note", "write_file", "run_python", "done"]
    assert state.history[2]["result"]["stdout"] == "hi\n"
    replayed = records(tmp_path / "replayed")[-1]
    assert replayed["total_steps"] == 4 and replayed["replayed_from"]["original_timestamp"] == entry["timestamp"]


def test_format_1_records_still_replay_and_summarize(tmp_path):
    """Logs written before format 2 (with a final_state) go through upgrade()."""
    import json

    from chem_llm.main import main
    from chem_llm.runlog import history_of, upgrade

    v1 = {"run_id": "old", "timestamp": "2026-10-07T00:00:00", "model": "Qwen", "work_dir": str(tmp_path / "old"),
          "runtime": 60.0, "num_steps": 2, "completed": True,
          "config": {"backend": "qwen", "decoding": {"do_sample": True, "temperature": 0.3, "max_new_tokens": 5000}},
          "code": {"commit": "abc", "dirty": ["chem_llm/agent.py", "bench/index.jsonl"]},
          "final_state": {"task": "the task", "final_result": "ok",
                          "history": [{"step": 1, "tool": "write_file", "args": {"path": "a.py", "content": "print(1)"},
                                       "result": "ok"},
                                      {"step": 2, "tool": "done", "args": {"summary": "ok"}, "result": "ok"}]}}
    rec = upgrade(v1)
    assert rec["format_version"] == 2 and rec["total_steps"] == 2 and rec["prompt"]["task"] == "the task"
    assert rec["config"]["temperature"] == 0.3 and "decoding" not in rec["config"]
    assert rec["config"]["code_commit"] == "abc" and rec["config"]["uncommitted_changes"] == ["chem_llm/agent.py"]
    assert [e["tool"] for e in history_of(v1)] == ["write_file", "done"]
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "log.jsonl").write_text(json.dumps(v1) + "\n")
    assert main(["--summarize-log", str(tmp_path / "old")]) == 0
    with mock_tools(run_espresso_workflow=fake_qe_workflow()):
        state = replay_run(v1, tmp_path / "again", verbose=False)
    assert (tmp_path / "again" / "a.py").read_text() == "print(1)" and state.done
