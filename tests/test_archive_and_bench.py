import json

import chem_llm.agent_core as agent_core
from chem_llm import bench
from chem_llm.archive import read_index
from conftest import HELLO


def test_every_run_is_archived(run_scripted, settings):
    result, _ = run_scripted(HELLO, label="smoke")
    d = result.archive_dir
    assert d.parent == settings.archive_root / "runs"
    assert "running hello.py" in (d / "progress.txt").read_text()
    assert (d / "summary.txt").read_text().startswith("chem_llm run summary")
    assert "First recorded run of Si" in (d / "changes.txt").read_text()
    record = json.loads((d / "record.json").read_text())
    assert list(record) == ["format_version", "run_id", "timestamp", "compound", "spacegroup", "model", "work_dir",
                            "completed", "total_steps", "runtime", "total_tokens", "outcome", "prompt", "steps",
                            "config", "metrics"]
    assert (record["compound"], record["spacegroup"], record["total_steps"]) == ("Si", "Fd-3m", 4)
    assert record["prompt"]["system"].startswith("You are") and record["prompt"]["task"]
    assert [s["tool"] for s in record["steps"]] == ["note", "write_file", "run_python", "done"]
    assert record["steps"][1]["args"] == {"path": "hello.py", "content": "print('hi')\n"}
    assert record["steps"][2]["result"]["stdout"] == "hi\n"
    assert record["total_tokens"] == sum(s["tokens"] for s in record["steps"]) > 0
    assert set(record["metrics"]) == {"operations", "phases", "n_tools", "qe_stages"}
    [entry] = read_index(settings.archive_root)
    assert entry["case"] == "Si" and entry["steps"] == 4 and entry["run_id"] == result.state.run_id
    assert entry["prompt_version"] and entry["total_tokens"] > 0
    assert (d / "scripts" / "hello.py").read_text() == "print('hi')\n"


def test_changes_since_previous_run(run_scripted, monkeypatch):
    run_scripted(HELLO)
    monkeypatch.setattr(agent_core, "SYSTEM_PROMPT_TEMPLATE", agent_core.SYSTEM_PROMPT_TEMPLATE + "\nNEW RULE: be brief.")
    second, _ = run_scripted(HELLO[1:])
    text = (second.archive_dir / "changes.txt").read_text()
    assert "Changes since the previous Si run" in text
    assert "+NEW RULE: be brief." in text and "1 line(s) added" in text
    assert "steps" in text and "4 -> 3" in text.replace("  ", " ").replace("  ", " ")


def test_evaluate_result_is_archived_and_indexed(run_scripted, settings):
    result, _ = run_scripted(HELLO, evaluate=lambda wd, rec: {"passed": False, "checks": [
        {"name": "band_gap", "ok": False, "detail": "x", "hard": True}]})
    assert json.loads((result.archive_dir / "checks.json").read_text())["passed"] is False
    entry = read_index(settings.archive_root)[-1]
    assert entry["passed"] is False and entry["failed_checks"] == ["band_gap"]


def test_archive_can_be_turned_off(run_scripted, settings):
    result, _ = run_scripted(HELLO, archive=False)
    assert result.archive_dir is None and not settings.archive_root.exists()


def test_suites_file_is_consistent():
    suites, cases = bench.load_suites()
    for s in suites.values():
        assert set(s["cases"]) <= set(cases)
    for name, c in cases.items():
        assert {"compound", "spacegroup", "purpose", "expect"} <= set(c), name
        exp = c["expect"]
        assert exp.get("metal") or "gap_eV" in exp, name
    assert bench.resolve(["core", "si", "tio2"])[-1] == "si"


def test_reference_comparison():
    ref = {"soc": False, "points": {"GAMMA": [-1.0, 0.0, 1.8], "X": [-2.0, -1.0, 2.5]}}
    ok = bench.compare_to_reference(ref, {"soc": False, "band_window": [-2, 1],
                                          "points": {"GAMMA": [-1.05, 0.0, 1.85], "X": [-2.0, -1.0, 2.5]}})
    assert ok["ok"] and "0.050" in ok["detail"]
    shifted = bench.compare_to_reference(ref, {"soc": False, "band_window": [-2, 1],
                                               "points": {"GAMMA": [-1.0, 0.0, 1.3], "X": [-2.0, -1.0, 2.5]}})
    assert not shifted["ok"] and "GAMMA" in shifted["detail"]
    assert not bench.compare_to_reference(ref, {"soc": True, "band_window": [0, 0], "points": {}})["ok"]
    assert "missing" in bench.compare_to_reference(ref, {"soc": False, "band_window": [0, 0],
                                                         "points": {"GAMMA": [0, 0, 0]}})["detail"]


def test_langsmith_scores_one_key_per_check():
    out = bench.langsmith_scores({"passed": True, "steps": 40, "wall_min": 6.1, "outcome": {"checks": [
        {"name": "band_gap", "ok": True, "detail": "1.8 eV"}, {"name": "plot", "ok": False, "detail": "none"}]}})
    scores = {r["key"]: r["score"] for r in out["results"]}
    assert scores == {"passed": 1, "band_gap": 1, "plot": 0, "steps": 40, "wall_min": 6.1}
    assert bench.langsmith_examples(["tio2"])[0]["inputs"]["compound"] == "TiO2"


def test_report_lists_every_case(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "BENCH", tmp_path)
    (tmp_path / "index.jsonl").write_text(json.dumps(
        {"case": "tio2", "time": "2026-10-13T10:00", "passed": True, "band_gap": 1.84, "steps": 40,
         "dir": "runs/x_tio2", "code": {"commit": "abc"}, "prompt_version": "123"}) + "\n")
    text = bench.write_report(tmp_path / "RESULTS.md").read_text()
    assert "| tio2 |" in text and "1.840 (1.7-2.0)" in text and "| si |" in text and "never" in text


def test_qe_output_reader(tmp_path):
    from chem_llm.qe_output import band_edges, read_pw_out

    soc = tmp_path / "soc.out"
    soc.write_text("     number of atoms/cell      =            5\n     number of electrons       =        82.00\n"
                   "     number of Kohn-Sham states=           82\n     Non magnetic calculation with spin-orbit\n"
                   "     highest occupied level (ev):     4.5905\n     convergence has been achieved in  12 iterations\n"
                   "   JOB DONE.\n")
    d = read_pw_out(soc)
    assert d["n_occupied"] == 82 and d["spin_orbit"] and d["highest_occupied_eV"] == 4.5905 and d["converged"]
    metal = tmp_path / "metal.out"
    metal.write_text("     number of electrons       =        16.00\n     the Fermi energy is     5.3839 ev\n"
                     "     total magnetization       =     2.21 Bohr mag/cell\n")
    d = read_pw_out(metal)
    assert d["n_occupied"] == 8 and d["fermi_energy_eV"] == 5.3839 and d["total_magnetization"] == 2.21
    e = band_edges([[0.0, 1.0, 3.0], [0.5, 2.0, 4.0]], 2)
    assert (e["vbm"], e["cbm"], e["gap"], e["vbm_k"]) == (2.0, 3.0, 1.0, 1)


def test_plot_axis_gives_plain_tick_lists(tmp_path):
    import numpy as np
    from ase.build import bulk

    from chem_llm.kpath import kp_save, plot_axis, standard_kpath

    kp = standard_kpath(bulk("Si"))
    kp_save(kp, tmp_path / "kpath.json")
    cell = np.asarray(kp["cell"]); recip = np.linalg.inv(cell).T
    k_cart = np.asarray(kp["kpoints"]) @ recip * np.linalg.norm(cell[0])     # 2pi/alat units
    n = len(k_cart)
    body = "\n".join(" ".join(f"{v:.6f}" for v in k) + "\n" + " ".join(["0.0"] * 4) for k in k_cart)
    (tmp_path / "b.dat").write_text(f" &plot nbnd=   4, nks= {n} /\n{body}\n")
    ax = plot_axis(tmp_path / "kpath.json", tmp_path / "b.dat")
    assert ax["tick_labels"][0] == "GAMMA" and len(ax["tick_x"]) == len(ax["ticks"])
    assert all(isinstance(x, float) for x in ax["tick_x"])


def test_qe_slots_limit_concurrency(monkeypatch, tmp_path):
    import threading
    import time

    from chem_llm.tools import _qe_slot

    monkeypatch.setattr("os.getuid", lambda: f"test{tmp_path.name}")
    inside, peak, lock = [0], [0], threading.Lock()

    def job():
        with _qe_slot(["a", "b"], {}):
            with lock:
                inside[0] += 1
                peak[0] = max(peak[0], inside[0])
            time.sleep(0.3)
            with lock:
                inside[0] -= 1

    threads = [threading.Thread(target=job) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert peak[0] == 2


def test_scf_kmesh_follows_the_reciprocal_lattice():
    from ase import Atoms

    from chem_llm.kpath import scf_kmesh

    rutile_like = Atoms("Ti", cell=[4.6, 4.6, 2.96], pbc=True)
    assert scf_kmesh(rutile_like) == [6, 6, 9]          # short c axis -> more points
    assert scf_kmesh(rutile_like, metal=True) == [10, 10, 15]


def test_dedicated_gpu_mode_pins_one_gpu_per_job(monkeypatch, tmp_path):
    import chem_llm.tools as tools
    from chem_llm.settings import Settings, activated

    seen = []
    monkeypatch.setattr("os.getuid", lambda: f"gpu{tmp_path.name}")
    monkeypatch.setattr(tools, "_run_qe_steps",
                        lambda wd, qe, s, steps, t, env=None: seen.append((qe.mpi_prefix, qe.pw_args, steps, env)) or {})
    s = Settings(work_root=tmp_path)
    s.qe.gpu_ids, s.qe.single_gpu_prefix, s.qe.pw_args = [2, 3], "srun -n 1", "-nk 4"
    (tmp_path / "calc").mkdir()
    with activated(s):
        tools.run_espresso_workflow(str(tmp_path / "calc"))
    prefix, pw_args, steps, env = seen[0]
    assert prefix == "srun -n 1 /usr/bin/env CUDA_VISIBLE_DEVICES=2" and pw_args == "" and env is None
    assert all(a == "" for _, _, a, _, _ in steps)


def test_validator_flags_a_spin_polarized_input_with_zero_moments(tmp_path):
    from chem_llm.tools.qe_validate import _check_magnetization

    fe = "&SYSTEM\n   nspin = 2\n   starting_magnetization = 0.5\n   starting_magnetization(1) = 0.0\n/\n"
    issues = _check_magnetization(fe)
    assert len(issues) == 2 and "non-magnetic" in issues[1]
    assert _check_magnetization("&SYSTEM\n   nspin = 2\n   starting_magnetization(1) = 0.5\n/\n") == []
    assert _check_magnetization("&SYSTEM\n   ecutwfc = 40\n/\n") == []


def test_ase_writes_moments_from_the_atoms(tmp_path):
    """The behaviour the magnetism rule relies on."""
    import io

    from ase.build import bulk
    from ase.io.espresso import write_espresso_in

    fe = bulk("Fe")
    fe.set_initial_magnetic_moments([0.5])
    out = io.StringIO()
    write_espresso_in(out, fe, input_data={"nspin": 2, "ecutwfc": 40}, pseudopotentials={"Fe": "Fe.upf"})
    assert "starting_magnetization(1) = 0.5" in out.getvalue()


def test_every_tool_dispatches_to_its_own_function():
    """Sweep 6: a helper inserted under validate_qe_input's @register_tool was
    registered as the tool, and every validation call crashed."""
    from chem_llm.tools import TOOL_DISPATCH, TOOLS

    for t in TOOLS:
        fn = TOOL_DISPATCH.get(t["name"])
        if fn is not None and getattr(fn, "__module__", "").startswith("chem_llm"):
            assert fn.__name__ == t["name"], (t["name"], fn.__name__)


def test_validate_qe_input_through_the_tool_table(tmp_path):
    from chem_llm.tools import TOOL_DISPATCH

    pw = tmp_path / "pw.in"
    pw.write_text("&SYSTEM\n   nat = 1\n   ntyp = 1\n   nspin = 2\n   starting_magnetization(1) = 0.0\n/\n"
                  "ATOMIC_SPECIES\nFe 55.8 Fe.upf\nATOMIC_POSITIONS crystal\nFe 0 0 0\n")
    result = TOOL_DISPATCH["validate_qe_input"](path=str(pw))
    assert result["success"] is False and any("non-magnetic" in i for i in result["issues"])


def test_validator_flags_heavy_elements_without_spin_orbit():
    from chem_llm.tools.qe_validate import _check_spin_orbit

    species = "ATOMIC_SPECIES\nCs 132.9 Cs.upf\nPb 207.2 Pb.pbesol-sr.UPF\nBr 79.9 Br.upf\n"
    assert "Cs, Pb" in _check_spin_orbit("&SYSTEM\n   nat = 5\n/\n" + species)[0]
    assert _check_spin_orbit("&SYSTEM\n   noncolin = .true.\n   lspinorb = .true.\n/\n" + species) == []
    assert _check_spin_orbit("&SYSTEM\n/\nATOMIC_SPECIES\nGa 69.7 Ga.upf\nAs 74.9 As.upf\n") == []
