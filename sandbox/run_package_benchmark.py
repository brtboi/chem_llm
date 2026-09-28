"""Run the packaged agent over a set of compounds.

    python sandbox/run_package_benchmark.py diamond
    python sandbox/run_package_benchmark.py diamond mgo gaas

One DFTAgent drives every compound in the list, so the local model is loaded
once for the whole sweep rather than once per compound -- which for a 27B
model is several minutes apiece. Each result line is appended to
sandbox/package_runs/results.jsonl, so several processes (or several
invocations after a node expires) report into the same place.

Backend comes from config.yaml; that is the local Qwen model, which runs on
the allocated GPU and costs nothing per step.
"""
import json
import sys
from pathlib import Path

from chem_llm import DFTAgent, Settings

# A spread of space groups and chemistries: cubic diamond-structure
# covalent, tetragonal oxide, cubic perovskite, rocksalt ionic,
# zincblende III-V, hexagonal wurtzite.
COMPOUNDS = {
    "diamond":  dict(compound="C",       spacegroup="Fd-3m",    spacegroup_number=227, spacegroup_name="diamond"),
    "tio2":     dict(compound="TiO2",    spacegroup="P4_2/mnm", spacegroup_number=136, spacegroup_name="rutile"),
    "cspbbr3":  dict(compound="CsPbBr3", spacegroup="Pm-3m",    spacegroup_number=221, spacegroup_name="cubic perovskite"),
    "mgo":      dict(compound="MgO",     spacegroup="Fm-3m",    spacegroup_number=225, spacegroup_name="rocksalt"),
    "gaas":     dict(compound="GaAs",    spacegroup="F-43m",    spacegroup_number=216, spacegroup_name="zincblende"),
    "zno":      dict(compound="ZnO",     spacegroup="P6_3mc",   spacegroup_number=186, spacegroup_name="wurtzite"),
    "si":       dict(compound="Si",      spacegroup="Fd-3m",    spacegroup_number=227, spacegroup_name="diamond-structure"),
}

SANDBOX = Path(__file__).resolve().parent
RESULTS = SANDBOX / "package_runs" / "results.jsonl"


def main(names):
    agent = DFTAgent(Settings.from_yaml(SANDBOX.parent / "config.yaml"))
    print(f"agent: {agent}", flush=True)
    agent.load()

    for name in names:
        spec = COMPOUNDS[name]
        print(f"\n{'=' * 70}\n=== {name}: {spec['compound']} ({spec['spacegroup']})\n{'=' * 70}", flush=True)
        try:
            result = agent.run(
                **spec,
                work_dir=agent.settings.work_root / name,
                log_file=SANDBOX / "logs" / f"package_{name}.jsonl",
            )
            record = dict(
                name=name, compound=spec["compound"], spacegroup=spec["spacegroup"],
                ok=result.ok, completed=result.completed, dft_finished=result.dft_finished,
                scf_converged=result.scf_converged, band_gap=result.band_gap,
                n_atoms=result.n_atoms, n_electrons=result.n_electrons,
                steps=result.num_steps, runtime_min=round(result.runtime_seconds / 60, 1),
                model=result.model, plot=str(result.plot) if result.plot else None,
                summary=result.summary,
            )
            print("\nRESULT:", result, flush=True)
        except Exception as e:  # keep going: one bad compound should not end the sweep
            record = dict(name=name, compound=spec["compound"], spacegroup=spec["spacegroup"],
                          ok=False, error=f"{type(e).__name__}: {e}")
            print(f"\nRESULT: {name} FAILED -- {type(e).__name__}: {e}", flush=True)

        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        with RESULTS.open("a") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:] or ["diamond"])
