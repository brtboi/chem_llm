import io
from types import SimpleNamespace

import pytest
from rich.console import Console

import chem_llm.main as cli
from chem_llm import vllm_server
from chem_llm.interactive import equivalent_command, wizard
from chem_llm.polymorphs import classify_spacegroup, group_by_spacegroup, parse_spacegroup
from chem_llm.tasks import build_task

QWEN = "Qwen/Qwen3.8-27B"
CACHED = [{"id": QWEN, "gb": 55.6, "path": "/x"}, {"id": "Qwen/Qwen3-30B-A3B-Instruct-2507", "gb": 61.1, "path": "/y"}]
TIO2 = [
    {"spacegroup_symbol": "I4_1/amd", "spacegroup_number": 141, "crystal_system": "tetragonal",
     "material_id": "mp-390", "nsites": 6, "energy_above_hull_eV_per_atom": 0.0, "band_gap_eV": 2.06, "observed": True},
    {"spacegroup_symbol": "P4_2/mnm", "spacegroup_number": 136, "crystal_system": "tetragonal",
     "material_id": "mp-2657", "nsites": 6, "energy_above_hull_eV_per_atom": 0.0095, "band_gap_eV": 1.77,
     "observed": True},
]


class Answers:
    """Scripted answers. Rich returns stream lines with their newline, which input() strips."""

    def __init__(self, answers):
        self.lines = iter(answers)

    def readline(self):
        return next(self.lines, "")


def run_wizard(settings, answers, *, compound=None, models=CACHED, served=(QWEN,), vllm="/usr/bin/vllm",
               downloads=None, lookup=lambda f, k: TIO2):
    ns = cli.build_parser().parse_args([compound] if compound else [])
    out = io.StringIO()
    result = wizard(ns, settings, console=Console(file=out, width=140), stream=Answers(answers), lookup=lookup,
                    models=models, served=lambda: None if served is None else list(served),
                    find_vllm=lambda: vllm,
                    downloader=(downloads.append if downloads is not None else lambda m: None))
    return result, out.getvalue()


def tail(path):  # model, serving, work dir, output, summary, advanced, start
    return ["1", "1", str(path), "1", "y", "n", "y"]


def test_typed_symbol_number_and_name(settings, tmp_path):
    args, out = run_wizard(settings, ["TiO2", "P4_2/mnm", "rutile"] + tail(tmp_path / "w"))
    assert (args.spacegroup, args.spacegroup_number, args.name) == ("P4_2/mnm", 136, "rutile")
    assert args.backend == "vllm" and not args.start_vllm and args.model is None
    assert args.work_dir == str(tmp_path / "w") and args.pretty_progress and args.summarize
    assert "chem-llm TiO2 P4_2/mnm --spacegroup-number 136 --name rutile --backend vllm" in out

    args, _ = run_wizard(settings, ["136", ""] + tail(tmp_path / "w"), compound="TiO2")
    assert (args.spacegroup, args.spacegroup_number, args.name) == ("P4_2/mnm", 136, "")

    # a phase name: no space group, and no second "common name" question
    args, out = run_wizard(settings, ["rutile"] + tail(tmp_path / "w"), compound="TiO2")
    assert (args.spacegroup, args.spacegroup_number, args.name) == (None, None, "rutile")
    assert "pick the rutile phase" in out and "chem-llm TiO2 rutile --backend vllm" in out


def test_search_materials_project(settings, tmp_path):
    args, out = run_wizard(settings, ["s", "1", ""] + tail(tmp_path / "w"), compound="TiO2")
    assert (args.spacegroup, args.spacegroup_number) == ("I4_1/amd", 141) and "mp-2657" in out
    # back to typing
    args, _ = run_wizard(settings, ["s", "3", "136", ""] + tail(tmp_path / "w"), compound="TiO2")
    assert args.spacegroup_number == 136


def test_search_failure_and_typo(settings, tmp_path):
    def broken(formula, key):
        raise ConnectionError("offline")

    args, out = run_wizard(settings, ["s", "P4_2/mmn", "n", "136", ""] + tail(tmp_path / "w"), compound="TiO2",
                           lookup=broken)
    assert args.spacegroup_number == 136 and "lookup failed" in out and "not a space group symbol" in out


def test_vllm_started_reused_or_missing(settings, tmp_path):
    args, out = run_wizard(settings, ["136", ""] + tail(tmp_path / "w"), compound="TiO2", served=None)
    assert args.backend == "vllm" and args.start_vllm and "--start-vllm" in out
    result, out = run_wizard(settings, ["136", ""] + tail(tmp_path / "w"), compound="TiO2", served=None, vllm=None)
    assert result is None and "vLLM not found" in out
    # a server is up but serves another model
    result, out = run_wizard(settings, ["136", ""] + tail(tmp_path / "w"), compound="TiO2", served=("other/m",))
    assert result is None and "serves ['other/m']" in out


def test_huggingface_serving_and_other_model(settings, tmp_path):
    args, _ = run_wizard(settings, ["136", "", "2", "2", str(tmp_path / "w"), "1", "y", "n", "y"],
                         compound="TiO2")
    assert args.model == "Qwen/Qwen3-30B-A3B-Instruct-2507" and args.backend == "huggingface"
    assert not args.start_vllm


def test_default_model_offered_for_download(settings, tmp_path, monkeypatch):
    monkeypatch.setattr("chem_llm.local_models.free_gb", lambda s: 500.0)
    got = []
    args, out = run_wizard(settings, ["136", "", "1", "y", "2", str(tmp_path / "w"), "1", "y", "n", "y"],
                           compound="TiO2", models=[], downloads=got)
    assert got == [QWEN] and "not downloaded" in out and args.backend == "huggingface"


def test_claude_needs_confirmation(settings, tmp_path):
    # Claude is option 3 (after the two local models); declining returns to the menu
    args, out = run_wizard(settings, ["136", "", "3", "n", "1", "2", str(tmp_path / "w"), "1", "y", "n", "y"],
                           compound="TiO2")
    assert args.backend == "huggingface" and "bills your Anthropic API credits" in out
    args, _ = run_wizard(settings, ["136", "", "3", "y", str(tmp_path / "w"), "1", "y", "n", "y"], compound="TiO2")
    assert args.backend == "claude"


def test_existing_work_dir_advanced_and_cancel(settings, tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    (work / "old.txt").write_text("x")
    args, _ = run_wizard(settings, ["136", "", "1", "1", str(work), "n", "3", "n", "y", "40", "10", "y"],
                         compound="TiO2")
    assert args.keep and args.quiet and (args.max_steps, args.structures) == (40, 10)
    assert "--keep --max-steps 40 --structures 10 --quiet" in equivalent_command(args)
    result, out = run_wizard(settings, ["136", ""] + tail(tmp_path / "x")[:-1] + ["n"], compound="TiO2")
    assert result is None and "Cancelled" in out


def test_classify_and_parse_spacegroup():
    assert parse_spacegroup("227") == ("Fd-3m", 227)
    assert classify_spacegroup("P42/mnm") == ("P4_2/mnm", 136, "")  # MP spelling
    assert classify_spacegroup("zinc blende") == (None, None, "zinc blende")
    for bad in ("0", "231"):
        with pytest.raises(ValueError):
            classify_spacegroup(bad)


def test_group_by_spacegroup_puts_observed_first():
    def doc(mid, sg, n, e, theoretical):
        return SimpleNamespace(material_id=mid, symmetry=SimpleNamespace(symbol=sg, number=n, crystal_system="Cubic"),
                               energy_above_hull=e, band_gap=0.0, nsites=2, theoretical=theoretical)

    found = group_by_spacegroup([doc("mp-1", "C2/m", 12, 0.0, True), doc("mp-66", "Fd-3m", 227, 0.11, False),
                                 doc("mp-9", "Fd-3m", 227, 0.05, True)])
    assert [f["spacegroup_symbol"] for f in found] == ["Fd-3m", "C2/m"]
    assert found[0]["material_id"] == "mp-9" and found[0]["observed"]  # most stable entry; group is observed


def test_task_for_a_phase_name_only():
    task = build_task("TiO2", spacegroup_name="rutile")
    assert "no space group was given: the phase is rutile" in task and "space group None" not in task
    # unchanged for a space group
    assert "generate_cif (space group P4_2/mnm, #136)" in build_task("TiO2", "P4_2/mnm", 136)


@pytest.fixture
def fake_agent(settings, monkeypatch):
    calls = []

    class Agent:
        def __init__(self, *a, **k):
            self.backend, self.docs = "huggingface", SimpleNamespace(available=True, path="x")

        def run(self, compound, spacegroup, **kw):
            calls.append((compound, spacegroup, kw["spacegroup_number"], kw["spacegroup_name"]))
            return SimpleNamespace(ok=True, work_dir="w", plot=None)

    monkeypatch.setattr(cli, "DFTAgent", Agent)
    monkeypatch.setattr(cli.Settings, "discover", classmethod(lambda cls: settings))
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    return calls


def test_cli_positional_is_symbol_number_or_name(fake_agent):
    for given, expected in (("P4_2/mnm", ("P4_2/mnm", 136, "")), ("136", ("P4_2/mnm", 136, "")),
                            ("rutile", (None, None, "rutile"))):
        cli.main(["TiO2", given, "--quiet"])
        assert fake_agent[-1] == ("TiO2", *expected)


def test_cli_without_a_terminal_does_not_prompt(fake_agent, capsys):
    """Scripts and pipes keep the old behaviour: an error, never a prompt."""
    with pytest.raises(SystemExit):
        cli.main([])
    assert "compound and spacegroup are required" in capsys.readouterr().err


def test_cli_interactive_cancel(fake_agent, monkeypatch):
    monkeypatch.setattr("chem_llm.interactive.wizard", lambda args, s: None)
    assert cli.main(["-i"]) == 1


def test_vllm_gpu_plan_and_command(settings, monkeypatch):
    monkeypatch.setattr(vllm_server, "visible_gpus", lambda: [0, 1, 2, 3])
    settings.qe.gpu_ids = [2, 3]
    assert vllm_server.plan_gpus(settings) == ([0, 1], 0.90)
    settings.qe.gpu_ids = [3]
    assert vllm_server.plan_gpus(settings) == ([0, 1], 0.90)  # 3 left -> TP 2
    settings.qe.gpu_ids = None
    assert vllm_server.plan_gpus(settings) == ([0, 1, 2, 3], 0.75)

    monkeypatch.setattr(vllm_server, "find_vllm", lambda s: "/opt/vllm/bin/vllm")
    argv, env, gpus = vllm_server.command(settings, QWEN, {"vision_config": {}, "text_config": {
        "max_position_embeddings": 262144}})
    assert argv[:3] == ["/opt/vllm/bin/vllm", "serve", QWEN] and "--limit-mm-per-prompt" in argv
    assert argv[argv.index("--max-model-len") + 1] == "131072" and env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"

    monkeypatch.setattr(vllm_server, "find_vllm", lambda s: None)
    with pytest.raises(vllm_server.VLLMError, match="vLLM not found"):
        vllm_server.command(settings, QWEN)
