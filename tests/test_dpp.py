"""The DeePseudopot leg: dpp_io bookkeeping and the second agent's loop (scripted model, training mocked)."""
import json
from pathlib import Path

import numpy as np
import pytest

from chem_llm import DFTAgent
from chem_llm.dpp.agent import DeePseudopotAgent, system_prompt
from chem_llm.dpp.io import (best_zunger_params, initial_zunger_params, pseudo_orbitals, read_fit, read_qe_run,
                             write_bundle, zunger_form)
from chem_llm.testing import ScriptedChatModel
from conftest import records

QE_RUN = Path(__file__).resolve().parents[1] / "deepseudopot_example" / "tio2_rutile" / "qe_run"
NN_CONFIG = {"num_cores": 0, "SObool": 0, "max_num_epochs": 100, "plotEvery": 50, "init_Zunger_num_epochs": 0}
INPUT = {"maxKE": 4.0, "idxVB": 15, "idxCB": 16, "systemName": "TiO2"}

pytestmark = pytest.mark.skipif(not QE_RUN.is_dir(), reason="example TiO2 QE run not in this checkout")


def bundle(out, run, **overrides):
    kw = dict(first_band=8, n_bands=26, kpoint_indices=[0, 26, 52], band_weights=[1.0] * 26,
              input_params=INPUT, nn_config=NN_CONFIG, init_params={"Ti": [0.0] * 9, "O": [0.0] * 9},
              energy_shift=-run["vbm"])
    kw.update(overrides)
    return write_bundle(out, run, **kw)


def test_read_qe_run_and_orbitals():
    run = read_qe_run(QE_RUN)
    assert run["symbols"].count("Ti") == 2 and run["n_electrons"] == 48 and run["n_occupied"] == 24
    assert not run["spin_orbit"] and run["energies"].shape[0] == len(run["kpoints"])
    assert abs(run["gap"] - 1.847) < 0.01
    assert [o["label"] for o in pseudo_orbitals(run["pseudopotentials"]["Ti"])] == ["3S", "3P", "3D", "4S"]
    # the semicore Ti 3s/3p bands (8 of them) lie far below O 2s
    E = run["energies"]
    assert E[:, 7].max() < E[:, 8].min() - 10


def test_write_bundle_files_and_alignment(tmp_path):
    run = read_qe_run(QE_RUN)
    summary = bundle(tmp_path / "b", run)
    files = set(summary["files"])
    assert {"system_0.par", "kpoints_0.par", "expBandStruct_0.par", "bandWeights_0.par", "input_0.par",
            "NN_config.par", "init_TiParams.par", "init_OParams.par"} <= files
    ref = np.loadtxt(tmp_path / "b" / "expBandStruct_0.par")
    assert ref.shape == (3, 27)  # path coordinate + 26 bands
    # column 1 is DFT band 8 (O 2s) at Gamma, shifted so the VBM is 0
    assert ref[0, 1] == pytest.approx(run["energies"][0, 8] - run["vbm"], abs=1e-5)
    assert "nBands = 26" in (tmp_path / "b" / "input_0.par").read_text()
    assert summary["kpoint_labels"] == ["GAMMA", "X", "M"]
    cell = (tmp_path / "b" / "system_0.par").read_text().splitlines()[3]
    assert float(cell.split()[0]) == pytest.approx(run["cell_angstrom"][0][0] / 0.529177210903, rel=1e-6)


@pytest.mark.parametrize("override, message", [
    ({"nn_config": {**NN_CONFIG, "num_cores": 2}}, "num_cores must be 0"),
    ({"nn_config": {**NN_CONFIG, "SObool": 1}}, "SObool = 1"),
    ({"band_weights": [1.0] * 5}, "band_weights has 5"),
    ({"first_band": 40}, "outside the 48 DFT bands"),
    ({"init_params": {"Ti": [0.0] * 9}}, "init_params must have exactly"),
    ({"input_params": {**INPUT, "idxCB": 30}}, "idxCB = 30"),
])
def test_write_bundle_rejects_inconsistent_bundles(tmp_path, override, message):
    with pytest.raises(ValueError, match=message):
        bundle(tmp_path / "b", read_qe_run(QE_RUN), **override)


def fake_training(offsets={"zunger_mc": 1.0, "gradient": 0.1}):
    """train_deepseudopot stand-in writing each stage's real output layout; the
    'fit' is the reference shifted by that stage's offset."""
    def train(inputs_folder, results_folder, mode="gradient", timeout_seconds=5400):
        inp, res = Path(inputs_folder), Path(results_folder)
        res.mkdir(parents=True, exist_ok=True)
        (res / "inputs_folder.txt").write_text(str(inp.resolve()) + "\n")
        ref = np.loadtxt(inp / "expBandStruct_0.par")
        ref[:, 1:] += offsets[mode]
        if mode == "zunger_mc":
            assert (inp / "mcOpts1.par").exists()
            np.savetxt(res / "bestBandStruct_57_0.dat", ref)
            for el in ("Ti", "O"):
                (res / f"best_{el}Params.dat").write_text("".join(f"{v}\n" for v in [0.2, 20.0, 1.1, 0.1, 0, 0, 0, 0, 0]))
            (res / "iterations.dat").write_text("newMSE \t curMSE \t bstMSE \t %Accpt\n0\t 9.0\t 9.0\t 1\n"
                                                "1 8.0\t 8.0\t 8.0\t 1\n2 8.5\t 8.0\t 8.0\t 0.5\n")
            return {"success": True, "mode": mode}
        np.savetxt(res / "epoch_100_BS_sys0.dat", ref)
        for el in ("Ti", "O"):
            (res / f"final_pot_{el}.dat").write_text("0 0\n")
        return {"success": True, "mode": mode, "final_pot_files": [str(p) for p in res.glob("final_pot*")],
                "trained_model_path": str(res / "epoch_100_PPmodel.pth")}
    return train


BUILD = f"""
from chem_llm.dpp.io import read_qe_run, write_bundle, initial_zunger_params, best_zunger_params
import sys
run = read_qe_run('qe_run')
stage = sys.argv[1] if len(sys.argv) > 1 else 'mc'
if stage == 'mc':
    init = {{el: d['params'] for el, d in initial_zunger_params(run, {{'Ti': ['3S', '3P']}}).items()}}
else:
    init = best_zunger_params('mc_results')
write_bundle(stage + '_inputs', run, first_band=8, n_bands=26, kpoint_indices=[0, 26, 52], band_weights=[1.0] * 26,
             input_params={INPUT!r}, nn_config={NN_CONFIG!r}, init_params=init, energy_shift=-run['vbm'],
             mc_opts={{'totalIter': 50}} if stage == 'mc' else None)
"""
BUILD_GD = BUILD.replace("sys.argv[1] if len(sys.argv) > 1 else 'mc'", "'gd'")


def test_agent_loop_with_its_own_prompt_and_tools(settings, tmp_path, monkeypatch):
    from chem_llm import tools

    schema, _ = tools.REGISTRY["train_deepseudopot"]
    monkeypatch.setitem(tools.REGISTRY, "train_deepseudopot", (schema, fake_training()))
    replies = [
        {"tool": "write_file", "args": {"path": "build_mc_inputs.py", "content": BUILD}},
        {"tool": "run_python", "args": {"path": "build_mc_inputs.py"}},
        {"tool": "generate_cif", "args": {"composition": "TiO2", "output_path": "x.cif"}},  # not this agent's
        {"tool": "train_deepseudopot", "args": {"inputs_folder": "mc_inputs", "results_folder": "mc_results",
                                                "mode": "zunger_mc"}},
        {"tool": "write_file", "args": {"path": "build_gd_inputs.py", "content": BUILD_GD}},
        {"tool": "run_python", "args": {"path": "build_gd_inputs.py"}},
        {"tool": "train_deepseudopot", "args": {"inputs_folder": "gd_inputs", "results_folder": "gd_results",
                                                "mode": "gradient"}},
        {"tool": "done", "args": {"summary": "trained"}},
    ]
    llm = ScriptedChatModel(replies=replies)
    agent = DeePseudopotAgent(DFTAgent(settings, llm=llm))
    result = agent.run(QE_RUN, tmp_path / "dpp", verbose=False)

    # the best stage is reported: gradient (0.1 eV) over Monte Carlo (1.0 eV)
    assert result.ok and result.fit["rms_error_eV"] == pytest.approx(0.1, abs=1e-4)
    assert result.fit["stage"] == "gradient" and result.results_dir.name == "gd_results"
    # the gradient stage started from the Monte Carlo stage's best parameters
    assert (tmp_path / "dpp" / "gd_inputs" / "init_TiParams.par").read_text().split()[:2] == ["0.2", "20.0"]
    assert result.fit["gap_fit_eV"] == pytest.approx(result.fit["gap_reference_eV"], abs=1e-4)
    history = result.state.history
    assert "ERROR executing generate_cif" in str(history[2]["result"])  # outside its tool table
    # the model saw the DeePseudopot system prompt, and the log records it
    assert "DeePseudopot fits its LOWEST nBands" in llm.seen[0][0].content
    assert "Monte Carlo first, then gradient descent" in llm.seen[0][0].content
    record = records(tmp_path / "dpp")[-1]
    assert record["prompt"]["system"] == system_prompt() and record["outcome"]["fit"]["rms_error_eV"] < 0.2
    assert (tmp_path / "dpp" / "qe_run" / "pw.out").exists()


def test_system_prompt_lists_only_its_tools():
    names = {t["name"] for t in json.loads(system_prompt().split("Available tools:\n", 1)[1])}
    assert names == {"read_file", "write_file", "run_python", "make_dir", "train_deepseudopot", "note", "done"}


def test_read_fit_needs_an_evaluated_epoch(tmp_path):
    (tmp_path / "res").mkdir()
    with pytest.raises(FileNotFoundError, match="epoch_.*bestBandStruct"):
        read_fit(tmp_path / "res", tmp_path)


def test_initial_zunger_params_from_the_upf_files():
    """Ti with its 3s/3p shells counted as core is a +4 ion, O a +6 ion; both screened by the
    32 electrons left in the fitted bands. The Zunger form fits each to within a fraction of the depth."""
    run = read_qe_run(QE_RUN)
    init = initial_zunger_params(run, {"Ti": ["3S", "3P"]})
    assert init["Ti"]["z_eff"] == 4 and init["O"]["z_eff"] == 6
    # Thomas-Fermi limit: v(0) = -4 pi z_eff / k_TF^2, so the ratio is z_eff's
    assert init["O"]["v0"] / init["Ti"]["v0"] == pytest.approx(6 / 4, rel=1e-3)
    assert -40 < init["Ti"]["v0"] < -20
    for d in init.values():
        assert len(d["params"]) == 9 and d["params"][4:] == [0.0] * 5
        assert d["fit_rms"] < 0.02 * abs(d["v0"])
        assert zunger_form(d["params"], 0.0) < 0 and abs(zunger_form(d["params"], 8.0)) < 1.0
    with pytest.raises(ValueError, match="not among the UPF orbitals"):
        initial_zunger_params(run, {"O": ["1S"]})


def test_mc_stage_files_and_read_fit(tmp_path):
    run = read_qe_run(QE_RUN)
    bundle(tmp_path / "in", run, mc_opts={"totalIter": 500, "tempStepSizeMod": [1.0, 3.0]})
    opts = (tmp_path / "in" / "mcOpts1.par").read_text()
    # DeePseudopot's read_mc_opts: 'key = a b', no commas
    assert "totalIter = 500\n" in opts and "tempStepSizeMod = 1.0 3.0\n" in opts and "," not in opts
    with pytest.raises(ValueError, match="two values"):
        bundle(tmp_path / "bad", run, mc_opts={"betas": [20]})
    res = tmp_path / "res"
    fake_training()(str(tmp_path / "in"), str(res), mode="zunger_mc")
    fit = read_fit(res, tmp_path / "in")
    assert fit["stage"] == "zunger_mc" and fit["rms_error_eV"] == pytest.approx(1.0, abs=1e-4)
    assert fit["mc_iterations"] == 2 and fit["mc_best_cost_by_quarter"][0] == 9.0 and fit["mc_best_cost_by_quarter"][-1] == 8.0
    assert best_zunger_params(res)["Ti"][:2] == [0.2, 20.0]


def test_training_tool_reads_elements_from_the_bundle(tmp_path):
    from chem_llm.tools.deepseudopot import _elements, train_deepseudopot

    bundle(tmp_path / "in", read_qe_run(QE_RUN))
    assert _elements(tmp_path / "in" / "system_0.par") == ["Ti", "O"]
    out = train_deepseudopot(str(tmp_path / "in"), str(tmp_path / "r"), mode="zunger_mc")
    assert not out["success"] and "mcOpts1.par" in out["stderr_tail"]  # no mc_opts were written
    assert not train_deepseudopot(str(tmp_path / "in"), str(tmp_path / "r"), mode="adam")["success"]
