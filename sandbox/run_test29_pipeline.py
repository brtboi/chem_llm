"""test29: rutile TiO2 with the packaged agent and current defaults.

    python sandbox/run_test29_pipeline.py
    python sandbox/check_tio2.py sandbox/test29

What's new relative to test27 (the last Qwen TiO2 run): trim_history on
(the default), read_file line ranges (start_line/num_lines), no grep tool,
run_python exit codes, and the INVARIANT 3 k-path fix. Local Qwen backend.

Second run here (after the CsPbBr3 failure in test29/cspbbr3): no
relativistic default anywhere -- the agent decides scalar vs fully
relativistic itself; get_pseudopotential has only required parameters and
reports z_valence etc. read from the downloaded file, with cutoffs from the
matching Pseudo-Dojo table; the harness refuses consecutive notes and a
repeated search_docs query. clear_dir keeps every *.jsonl log, including
test29/cspbbr3/log.jsonl, so the earlier runs stay on record.
"""
from pathlib import Path

from chem_llm import DFTAgent

SANDBOX = Path(__file__).resolve().parent

if __name__ == "__main__":
    agent = DFTAgent(SANDBOX.parent / "config.yaml")
    print(agent, "| trim_history:", agent.settings.agent.trim_history, flush=True)
    result = agent.run(
        "TiO2", "P4_2/mnm",
        spacegroup_number=136,
        spacegroup_name="rutile",
        work_dir=SANDBOX / "test29",
        clear_dir=True,
    )
    print("\nRESULT:", result, flush=True)
