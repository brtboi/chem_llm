"""The agent loop end to end: scripted model, real tools (QE mocked)."""
import io

from chem_llm.progress import ProgressPrinter
from conftest import HELLO, records


def test_run_completes_and_executes_tools(run_scripted, tmp_path):
    result, llm = run_scripted(HELLO)
    assert result.completed and result.num_steps == 4
    assert [e["tool"] for e in result.state.history] == ["note", "write_file", "run_python", "done"]
    run = result.state.history[2]["result"]
    assert run["stdout"] == "hi\n" and run["returncode"] == 0
    assert (tmp_path / "run" / "hello.py").exists()
    assert len(llm.seen) == 4


def test_prompt_is_system_prompt_plus_working_memory(run_scripted):
    from chem_llm.agent_core import SYSTEM_PROMPT_TEMPLATE

    result, llm = run_scripted(HELLO)
    system, user = llm.seen[2]
    assert system.type == "system" and system.content == SYSTEM_PROMPT_TEMPLATE
    assert user.type == "human" and user.content.startswith("Task: ")
    assert "hello.py" in user.content  # the write at step 2 is in its memory by step 3


def test_steps_are_timed_and_traced(run_scripted, tmp_path):
    result, _ = run_scripted(HELLO)
    steps = result.state.steps
    assert [s["tool"] for s in steps] == ["note", "write_file", "run_python", "done"]
    for s in steps:
        assert s["llm_s"] >= 0 and s["tool_s"] >= 0 and s["t_end"] >= s["t_start"]
        assert s["prompt_tokens"] > 0 and s["completion_tokens"] > 0
    trace = records(tmp_path / "run", "trace.jsonl")
    assert len(trace) == 4 and {t["run_id"] for t in trace} == {result.state.run_id}
    assert trace[2]["entries"][0]["tool"] == "run_python"


def test_log_record_has_prompt_metrics_and_outcome(run_scripted, tmp_path):
    result, _ = run_scripted(HELLO)
    rec = records(tmp_path / "run")[-1]
    assert rec["run_id"] == result.state.run_id
    assert len(rec["prompt"]["version"]) == 12 and rec["prompt"]["system"]
    m = rec["metrics"]
    assert set(m["phases_s"]) == {"llm", "run_code", "qe", "other_tools"}
    assert m["llm"]["calls"] == 4 and m["llm"]["prompt_tokens"]["max"] > 0
    assert m["n_tools"] == {"note": 1, "write_file": 1, "run_python": 1, "done": 1}
    assert "band_gap" in rec["outcome"]
    assert rec["config"]["backend"] == "custom"
    assert result.metrics == m


def test_summary_has_timing_section(run_scripted):
    result, _ = run_scripted(HELLO)
    text = result.report()
    assert "5. Where the time went" in text and "model generation" in text


def test_two_runs_append_to_the_same_log(run_scripted, tmp_path):
    first, _ = run_scripted(HELLO)
    second, _ = run_scripted(HELLO)
    recs = records(tmp_path / "run")
    assert [r["run_id"] for r in recs] == [first.state.run_id, second.state.run_id]
    trace = records(tmp_path / "run", "trace.jsonl")
    assert len(trace) == 8


def test_back_to_back_notes_are_refused(run_scripted):
    result, _ = run_scripted([
        {"tool": "note", "args": {"text": "a"}},
        {"tool": "note", "args": {"text": "b"}},
        {"tool": "done", "args": {"summary": "x"}},
    ])
    assert [e["tool"] for e in result.state.history] == ["note", "refused", "done"]
    assert result.state.steps[1].get("refused") is True


def test_unparseable_reply_is_fed_back_and_the_run_continues(run_scripted):
    result, _ = run_scripted(["I think I should write a file.", {"tool": "done", "args": {"summary": "x"}}])
    assert [e["tool"] for e in result.state.history] == ["parse_error", "done"]
    assert result.completed
    assert any("failed to parse" in n for n in result.state.scratchpad)


def test_fenced_json_reply_is_accepted(run_scripted):
    result, _ = run_scripted(['```json\n{"tool": "done", "args": {"summary": "x"}}\n```'])
    assert result.completed


def test_max_steps_stops_the_run(run_scripted):
    replies = [{"tool": "make_dir", "args": {"path": f"d{i}"}} for i in range(10)]
    result, _ = run_scripted(replies, max_steps=3)
    assert not result.completed and result.num_steps == 3
    assert "Max steps reached" in result.state.scratchpad[-1]


def test_model_errors_end_the_run_after_three_in_a_row(run_scripted):
    result, _ = run_scripted([ConnectionError("server down")] * 5)
    assert [e["tool"] for e in result.state.history] == ["api_error"] * 3
    assert not result.completed


def test_mocked_qe_reports_stage_timings(run_scripted):
    result, _ = run_scripted([
        {"tool": "make_dir", "args": {"path": "calculations/000"}},
        {"tool": "run_espresso_workflow", "args": {"directory": "calculations/000"}},
        {"tool": "done", "args": {"summary": "x"}},
    ])
    assert result.metrics["qe_stages_s"] == {"scf": 1.0, "bands_scf": 2.0, "bands_post": 0.5}
    assert "qe" in result.metrics["phases_s"] and result.metrics["n_tools"]["run_espresso_workflow"] == 1


def test_progress_observer_sees_every_step(run_scripted):
    out = io.StringIO()
    result, _ = run_scripted(HELLO, on_step=ProgressPrinter(out))
    text = out.getvalue()
    assert "note: Plan" in text and "writing hello.py" in text and "running hello.py" in text and "done:" in text
