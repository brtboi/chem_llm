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
    assert records(tmp_path / "replayed")[-1]["metrics"]["llm"]["calls"] == 4
