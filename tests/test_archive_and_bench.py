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
    assert "final_state" not in record and "system" not in record["prompt"] and record["label"] == "smoke"
    [entry] = read_index(settings.archive_root)
    assert entry["case"] == "Si" and entry["steps"] == 4 and entry["run_id"] == result.state.run_id
    assert (settings.archive_root / "prompts" / f"{entry['system_sha']}.txt").exists()


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
        with _qe_slot(2, {}):
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
