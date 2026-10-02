"""test29 second compound: cubic CsPbBr3 (Pm-3m), as different from rutile TiO2
as the pipeline allows -- simple cubic instead of tetragonal, heavy Pb/Cs
(tests the relativistic pseudopotential choice), and a soft halide lattice
(tests the thermal-displacement choice).

    python sandbox/run_test29_cspbbr3.py

Runs in sandbox/test29/cspbbr3 so the TiO2 run in sandbox/test29 is kept.
Current defaults (trim_history on, relativistic left at the task default),
local Qwen backend.
"""
from pathlib import Path

from chem_llm import DFTAgent

SANDBOX = Path(__file__).resolve().parent

if __name__ == "__main__":
    agent = DFTAgent(SANDBOX.parent / "config.yaml")
    print(agent, "| trim_history:", agent.settings.agent.trim_history, flush=True)
    result = agent.run(
        "CsPbBr3", "Pm-3m",
        spacegroup_number=221,
        spacegroup_name="cubic perovskite",
        work_dir=SANDBOX / "test29" / "cspbbr3",
        clear_dir=True,
    )
    print("\nRESULT:", result, flush=True)
