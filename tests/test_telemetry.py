from chem_llm.state import AgentState
from chem_llm.telemetry import format_timing, run_metrics

TASK = "STEP 1: Fetch the base structure. STEP 2: Write and run generate_structures.py. STEP 3: Fetch pseudopotentials."


def _state(entries, steps):
    s = AgentState(TASK)
    for tool, args, result in entries:
        s.log(tool, args, result)
    s.steps = steps
    return s


def test_phases_tools_and_tokens():
    s = _state(
        [("generate_cif", {}, {}), ("run_python", {"path": "a.py"}, {"returncode": 0}),
         ("run_espresso_workflow", {"directory": "c"}, {"success": True, "timings_s": {"scf": 20.0, "bands_scf": 70.0}})],
        [{"step": 1, "tool": "generate_cif", "llm_s": 40.0, "tool_s": 5.0, "prompt_tokens": 9000, "completion_tokens": 50, "history": [1], "t_end": 45},
         {"step": 2, "tool": "run_python", "llm_s": 50.0, "tool_s": 10.0, "prompt_tokens": 11000, "completion_tokens": 450, "history": [2], "t_end": 105},
         {"step": 3, "tool": "run_espresso_workflow", "llm_s": 10.0, "tool_s": 90.0, "prompt_tokens": 12000, "completion_tokens": 0, "history": [3], "t_end": 205}],
    )
    m = run_metrics(s, wall_s=210.0, model_load_s=120.0)
    assert m["phases_s"] == {"llm": 100.0, "materials_project": 5.0, "agent_scripts": 10.0,
                             "quantum_espresso": 90.0, "harness": 5.0}
    assert m["llm"]["prompt_tokens"] == {"total": 32000, "mean": 10667, "max": 12000}
    assert m["llm"]["completion_tokens_per_s"] == 5.0
    assert m["qe_stages_s"] == {"scf": 20.0, "bands_scf": 70.0}
    assert list(m["tools"])[0] == "run_espresso_workflow"  # slowest first
    text = "\n".join(format_timing(m))
    assert "model generation" in text and "Quantum ESPRESSO" in text and "loading the model" in text


def test_workflow_stages_from_checkpoint_notes():
    s = _state(
        [("generate_cif", {}, {}), ("note", {"text": "STEP 1 done: fetched mp-149"}, "noted"),
         ("run_python", {"path": "g.py"}, {"returncode": 0}),
         ("note", {"text": "STEP 2 checkpoint: 51 structures verified"}, "noted"),
         ("note", {"text": "STEP 1 done again (should be ignored)"}, "noted")],
        [{"step": i, "tool": t, "llm_s": 1.0, "tool_s": 0.0, "history": [i], "t_end": 10.0 * i}
         for i, t in enumerate(["generate_cif", "note", "run_python", "note", "note"], 1)],
    )
    stages = run_metrics(s, wall_s=50.0)["workflow_stages"]
    assert [(st["stage"], st["duration_s"], st["ended_at_step"]) for st in stages] == [(1, 20.0, 2), (2, 20.0, 4)]
    assert stages[0]["title"].startswith("Fetch the base structure")


def test_old_runs_without_timing():
    assert "not recorded" in format_timing(None)[0]
