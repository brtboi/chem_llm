from chem_llm.state import AgentState
from chem_llm.telemetry import format_timing, run_metrics

TASK = "STEP 1: Fetch the base structure. STEP 2: Write and run generate_structures.py. STEP 3: Fetch pseudopotentials."


def _state(entries, steps):
    s = AgentState(TASK)
    for tool, args, result in entries:
        s.log(tool, args, result)
    s.steps = steps
    return s


def test_operations_phases_and_tool_counts():
    s = _state(
        [("generate_cif", {}, {}), ("run_python", {"path": "generate_structures.py"}, {"returncode": 0}),
         ("search_docs", {"query": "x"}, {}),
         ("run_espresso_workflow", {"directory": "c"}, {"success": True, "timings_s": {"scf": 20.0, "bands_scf": 70.0}})],
        [{"step": 1, "tool": "generate_cif", "llm_s": 40.0, "tool_s": 5.0, "prompt_tokens": 9000, "completion_tokens": 50, "history": [1]},
         {"step": 2, "tool": "run_python", "llm_s": 50.0, "tool_s": 10.0, "prompt_tokens": 11000, "completion_tokens": 450, "history": [2]},
         {"step": 3, "tool": "search_docs", "llm_s": 5.0, "tool_s": 2.0, "prompt_tokens": 11500, "completion_tokens": 20, "history": [3]},
         {"step": 4, "tool": "run_espresso_workflow", "llm_s": 10.0, "tool_s": 90.0, "prompt_tokens": 12000, "completion_tokens": 0, "history": [4]}],
    )
    m = run_metrics(s, wall_s=215.0, model_load_s=120.0)
    ops = m["operations"]
    assert ops["llm"] == {"time": 105.0, "tokens": 44020}
    assert ops["tools"] == {"time": 7.0, "tokens": 9050 + 11520}
    assert ops["code"] == {"time": 10.0, "tokens": 11450}
    assert ops["qe"] == {"time": 90.0, "tokens": 12000}
    assert ops["total"] == {"time": 215.0, "tokens": 44020}
    # the doc search belongs to the step it prepares (QE, STEP 6)
    assert m["phases"]["1_structure"] == {"time": 45.0, "tokens": 9050}
    assert m["phases"]["2_perturbed_structures"] == {"time": 60.0, "tokens": 11450}
    assert m["phases"]["6_run_qe"] == {"time": 107.0, "tokens": 23520}
    assert m["n_tools"] == {"generate_cif": 1, "run_python": 1, "search_docs": 1, "run_espresso_workflow": 1}
    assert m["qe_stages"] == {"scf": 20.0, "bands_scf": 70.0}
    text = "\n".join(format_timing(m))
    assert "model generation" in text and "Quantum ESPRESSO" in text and "1 fetch + reduce structure" in text


def test_notes_name_their_workflow_step():
    from chem_llm.telemetry import assign_phases

    s = _state([("note", {"text": "STEP 4 decisions: ecutwfc ..."}, "noted"), ("read_file", {"path": "x.txt"}, {}),
                ("write_file", {"path": "plot_bands.py"}, {})],
               [{"step": i, "history": [i]} for i in (1, 2, 3)])
    assert assign_phases(s.history, s.steps) == [4, 7, 7]


def test_old_runs_without_timing():
    assert "not recorded" in format_timing(None)[0]
