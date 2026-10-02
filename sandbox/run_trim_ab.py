"""A/B of trim_history on diamond, local Qwen, one model load for both arms.

    python sandbox/run_trim_ab.py            # trim on, then trim off

Both arms use the same prompt and tools (ranged read_file, grep_file); only
agent.settings.agent.trim_history differs. Each arm writes to its own
work_dir and log file, and a result line (with "arm") is appended to
sandbox/package_runs/results.jsonl.
"""
import json
from pathlib import Path

from chem_llm import DFTAgent

SANDBOX = Path(__file__).resolve().parent
RESULTS = SANDBOX / "package_runs" / "results.jsonl"
SPEC = dict(compound="C", spacegroup="Fd-3m", spacegroup_number=227, spacegroup_name="diamond")


def main():
    agent = DFTAgent(SANDBOX.parent / "config.yaml")   # local Qwen backend
    print(f"agent: {agent}", flush=True)
    agent.load()

    for trim in (True, False):
        arm = f"trim_{'on' if trim else 'off'}"
        agent.settings.agent.trim_history = trim
        print(f"\n{'=' * 70}\n=== diamond, {arm}\n{'=' * 70}", flush=True)
        result = agent.run(
            **SPEC,
            work_dir=SANDBOX / "package_runs" / f"diamond_{arm}",
            log_file=SANDBOX / "logs" / f"package_diamond_{arm}.jsonl",
        )
        print("\nRESULT:", arm, result, flush=True)
        record = dict(
            name=f"diamond_{arm}", arm=arm, trim_history=trim, compound="C", spacegroup="Fd-3m",
            ok=result.ok, completed=result.completed, dft_finished=result.dft_finished,
            band_gap=result.band_gap, steps=result.num_steps,
            runtime_min=round(result.runtime_seconds / 60, 1), model=result.model,
            plot=str(result.plot) if result.plot else None,
        )
        with RESULTS.open("a") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
