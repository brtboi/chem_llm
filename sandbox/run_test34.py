"""test34: band structures for a spread of compounds, one agent, local Qwen.

    python sandbox/run_test34.py              # all, in order
    python sandbox/run_test34.py graphite     # a subset

Each compound gets its own directory, sandbox/test34/<name>/ (log.jsonl,
scripts, calculations/000/bands_000.pdf), plus:
    sandbox/logs/test34_<name>_progress.log   readable live stream
    sandbox/logs/test34_<name>_summary.txt    four-section run summary
and a line in sandbox/test34/results.jsonl. Runs never share a directory,
so nothing overwrites anything.
"""
import json
import sys
import traceback
from pathlib import Path

from chem_llm import DFTAgent
from chem_llm.progress import ProgressPrinter

SANDBOX = Path(__file__).resolve().parent
ROOT = SANDBOX / "test34"
LOGS = SANDBOX / "logs"

COMPOUNDS = {
    # semimetal (needs smearing), layered hexagonal
    "graphite": dict(compound="C", spacegroup="P6_3/mmc", spacegroup_number=194, spacegroup_name="graphite"),
    # heavy elements -> spin-orbit; the earlier plot bug
    "cspbbr3": dict(compound="CsPbBr3", spacegroup="Pm-3m", spacegroup_number=221, spacegroup_name="cubic perovskite"),
    # zincblende III-V, small direct gap
    "gaas": dict(compound="GaAs", spacegroup="F-43m", spacegroup_number=216, spacegroup_name="zincblende"),
    # layered transition-metal dichalcogenide, indirect gap
    "mos2": dict(compound="MoS2", spacegroup="P6_3/mmc", spacegroup_number=194, spacegroup_name="2H"),
    # polar wurtzite, no inversion centre
    "zno": dict(compound="ZnO", spacegroup="P6_3mc", spacegroup_number=186, spacegroup_name="wurtzite"),
}


def main(names):
    ROOT.mkdir(parents=True, exist_ok=True)
    agent = DFTAgent(SANDBOX.parent / "config.yaml")
    print(agent, flush=True)
    agent.load()

    for name in names:
        spec = COMPOUNDS[name]
        work_dir = ROOT / name
        print(f"=== {name}: {spec['compound']} ({spec['spacegroup']}) -> {work_dir}", flush=True)
        with open(LOGS / f"test34_{name}_progress.log", "w") as progress:
            try:
                result = agent.run(**spec, work_dir=work_dir, clear_dir=True,
                                   verbose=False, on_step=ProgressPrinter(progress))
                (LOGS / f"test34_{name}_summary.txt").write_text(result.report())
                record = dict(name=name, ok=result.ok, completed=result.completed, band_gap=result.band_gap,
                              steps=result.num_steps, runtime_min=round(result.runtime_seconds / 60, 1),
                              plot=str(result.plot) if result.plot else None)
            except Exception as e:  # one failure must not end the sweep
                traceback.print_exc(file=progress)
                record = dict(name=name, ok=False, error=f"{type(e).__name__}: {e}")
        print("   ", record, flush=True)
        with open(ROOT / "results.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:] or list(COMPOUNDS))
