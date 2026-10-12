from pathlib import Path

import chem_llm.agent_core as agent_core
from chem_llm.main import main
from chem_llm.rundiff import diff_runs, parse_run_spec, prompt_snapshot
from conftest import HELLO


def test_parse_run_spec():
    assert parse_run_spec("runs/zno") == (Path("runs/zno"), -1)
    assert parse_run_spec("runs/zno@0") == (Path("runs/zno"), 0)
    assert parse_run_spec("runs/zno/log.jsonl@-2") == (Path("runs/zno/log.jsonl"), -2)
    assert parse_run_spec("user@host/runs") == (Path("user@host/runs"), -1)


def test_prompt_version_tracks_system_prompt_and_task():
    a = prompt_snapshot("sys", "task")
    assert a["version"] == prompt_snapshot("sys", "task")["version"]
    assert a["version"] != prompt_snapshot("sys2", "task")["version"]
    assert a["version"] != prompt_snapshot("sys", "task2")["version"]


def test_diff_shows_prompt_change_and_cost(run_scripted, tmp_path, monkeypatch):
    run_scripted(HELLO)
    monkeypatch.setattr(agent_core, "SYSTEM_PROMPT_TEMPLATE",
                        agent_core.SYSTEM_PROMPT_TEMPLATE.replace("fewest necessary tool calls", "fewest tool calls"))
    run_scripted(HELLO[1:])
    text = diff_runs(f"{tmp_path / 'run'}@0", f"{tmp_path / 'run'}@1")
    assert "-Your objective is to complete the task using the fewest necessary tool calls" in text.replace("      ", "")
    assert "+Your objective is to complete the task using the fewest tool calls" in text.replace("      ", "")
    assert "(same prompt)" not in text
    steps = next(line for line in text.splitlines() if line.strip().startswith("steps"))
    assert steps.split()[1:] == ["4", "3", "*"]
    assert "task: identical" in text


def test_same_prompt_is_reported(run_scripted, tmp_path):
    run_scripted(HELLO)
    run_scripted(HELLO)
    text = diff_runs(f"{tmp_path / 'run'}@0", str(tmp_path / "run"))
    assert "(same prompt)" in text and "system prompt: identical" in text


def test_cli_diff_runs(run_scripted, tmp_path, capsys):
    run_scripted(HELLO)
    run_scripted(HELLO)
    assert main(["--diff-runs", f"{tmp_path / 'run'}@0", f"{tmp_path / 'run'}@1"]) == 0
    assert "chem_llm run comparison" in capsys.readouterr().out
    assert main(["--diff-runs", str(tmp_path / "missing"), str(tmp_path / "run")]) == 2


def test_cli_summarize_log_includes_timing(run_scripted, tmp_path, capsys):
    run_scripted(HELLO)
    assert main(["--summarize-log", str(tmp_path / "run")]) == 0
    assert "5. Where the time went" in capsys.readouterr().out
