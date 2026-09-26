"""Run the packaged pipeline over a set of compounds, one per process.

    python sandbox/run_package_benchmark.py diamond
    python sandbox/run_package_benchmark.py tio2 cspbbr3

Each compound is a single `run_dft_agent` call -- the whole point of the
package is that this is all it takes -- and the result line is appended to
sandbox/package_runs/results.jsonl so several processes can report into the
same place.
"""
import json
import sys
from pathlib import Path

from chem_llm import configure, run_dft_agent

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

RESULTS = Path("sandbox/package_runs/results.jsonl")


def main(names):
    configure("config.yaml")
    for name in names:
        spec = COMPOUNDS[name]
        print(f"\n{'=' * 70}\n=== {name}: {spec['compound']} ({spec['spacegroup']})\n{'=' * 70}", flush=True)
        try:
            result = run_dft_agent(run_name=name, **spec)
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
