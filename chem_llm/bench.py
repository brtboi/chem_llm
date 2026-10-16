"""Benchmark the agent on a fixed set of compounds and keep the history.

One place instead of numbered sandbox directories:

    bench/suites.yaml      every case: compound, what it tests, the physics
                           expected (gap range or metal, spin-orbit, magnetism)
    bench/references/      golden band energies at the high-symmetry points
                           of a validated run (`chem-llm-bench promote`)
    bench/index.jsonl      one line per run, ever (the run archive's index)
    bench/runs/<run>/      progress.txt, summary.txt, changes.txt, checks.json
    bench/RESULTS.md       generated: latest result per case + history
    bench/work/<case>/     scratch work dirs (not committed)

    chem-llm-bench list
    chem-llm-bench run core --parallel 3 --label "occupations rule v2"
    chem-llm-bench run zno gaas
    chem-llm-bench report
    chem-llm-bench promote bench/runs/2026-10-13_235901_zno

Each case runs in its own process (tools work relative to the process's
cwd), several at once against one vLLM server; Quantum ESPRESSO steps take
turns on the node's GPUs (QESettings.max_concurrent at a time). A run is scored by checks
read from the files it produced -- never from the agent's own claims.

With LANGSMITH_API_KEY set, `--langsmith` also records the sweep as a
LangSmith experiment on a dataset named after the suite, so experiments
(prompt versions, models) can be compared in its UI; each agent run is
traced there too.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from . import REPO_ROOT

BENCH = REPO_ROOT / "bench"
SUITES = BENCH / "suites.yaml"
REFERENCES = BENCH / "references"


# ----------------------------------------------------------------------
# cases
def load_suites(path: Path = SUITES) -> tuple[dict, dict]:
    data = yaml.safe_load(Path(path).read_text())
    return data.get("suites", {}), data.get("cases", {})


def resolve(names: list[str], path: Path = SUITES) -> list[str]:
    """Suite names expand to their cases; case names stay."""
    suites, cases = load_suites(path)
    out: list[str] = []
    for name in names:
        if name in suites:
            out += [c for c in suites[name]["cases"] if c not in out]
        elif name in cases:
            if name not in out:
                out.append(name)
        else:
            raise SystemExit(f"chem-llm-bench: no suite or case named {name!r} in {path}")
    return out


def work_dir_of(name: str, case: dict) -> Path:
    return (REPO_ROOT / case["work_dir"]) if case.get("work_dir") else BENCH / "work" / name


# ----------------------------------------------------------------------
# what a run produced
def _read(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


def calc_facts(work_dir: Path, job: str = "000") -> dict:
    """Physics facts from the QE files a run left in calculations/<job>."""
    from .pipeline import _inspect_outputs

    calc = Path(work_dir) / "calculations" / job
    facts = dict(_inspect_outputs(Path(work_dir), job=job))
    pw_in, pw_out = _read(calc / "pw.in") or "", _read(calc / "pw.out") or ""
    facts["soc"] = bool(re.search(r"lspinorb\s*=\s*\.?t", pw_in, re.I)) or "spin-orbit" in pw_out.lower()
    facts["smearing"] = bool(re.search(r"occupations\s*=\s*['\"]smearing", pw_in, re.I))
    facts["spin_polarized"] = bool(re.search(r"nspin\s*=\s*2", pw_in) or re.search(r"starting_magnetization", pw_in))
    m = re.findall(r"the Fermi energy is\s+([-\d.]+)\s*ev", pw_out)
    facts["fermi_eV"] = float(m[-1]) if m else None
    m = re.findall(r"total magnetization\s*=\s*([-\d.]+)", pw_out)
    facts["total_magnetization"] = float(m[-1]) if m else None
    nat = facts.get("n_atoms")
    if facts["total_magnetization"] is not None and nat:
        facts["magnetization_per_atom"] = round(abs(facts["total_magnetization"]) / nat, 3)
    return facts


def band_fingerprint(work_dir: Path, metal: bool = False, n_around: int = 4, job: str = "000") -> dict | None:
    """Band energies at each labelled high-symmetry point, relative to the
    VBM (or the Fermi energy for a metal): the n_around bands below and
    above the gap. None if the run has no kpath.json / bands output."""
    import numpy as np

    from .kpath import plot_axis

    calc = Path(work_dir) / "calculations" / job
    facts = calc_facts(work_dir, job)
    n_occ = facts.get("n_occupied")
    try:
        ax = plot_axis(calc / "kpath.json", calc / f"{job}.bands.dat")
    except (OSError, ValueError, KeyError):
        return None
    E = np.asarray(ax["energies"])
    if not n_occ or E.shape[1] <= n_occ:
        return None
    if metal:
        if facts.get("fermi_eV") is None:
            return None
        zero = facts["fermi_eV"]
    else:
        zero = float(E[:, n_occ - 1].max())
    lo, hi = max(0, n_occ - n_around), min(E.shape[1], n_occ + n_around)
    points = {}
    for tick in ax["ticks"]:
        for label, idx in zip(tick["label"].split("|"), tick["indices"]):
            points.setdefault(label, [round(float(e - zero), 4) for e in E[idx, lo:hi]])
    return {"zero": "fermi" if metal else "vbm", "soc": facts["soc"], "band_window": [lo - n_occ, hi - n_occ],
            "points": points, "path": ax["path"]}


# ----------------------------------------------------------------------
# checks
def _check(name: str, ok: bool | None, detail: str, hard: bool = True) -> dict:
    return {"name": name, "ok": ok, "detail": detail, "hard": hard}


def evaluate_case(name: str, case: dict, work_dir: Path, record: dict | None = None) -> dict:
    """Score one finished run of `case` from its files."""
    exp = case.get("expect") or {}
    facts = calc_facts(work_dir)
    metal = bool(exp.get("metal"))
    checks = [
        _check("completed", bool((record or {}).get("completed")), "agent called done"
               if (record or {}).get("completed") else "agent stopped without calling done"),
        _check("dft_finished", facts.get("dft_finished") and facts.get("scf_converged"),
               f"SCF converged: {facts.get('scf_converged')}, all three QE stages JOB DONE: {facts.get('dft_finished')}"),
        _check("plot", facts.get("plot") is not None, str(facts.get("plot") or "no plot in calculations/000")),
    ]
    gap = facts.get("band_gap")
    if metal:
        checks.append(_check("metal_occupations", facts["smearing"] and facts.get("fermi_eV") is not None,
                             f"smearing: {facts['smearing']}, Fermi energy: {facts.get('fermi_eV')}"))
    elif "gap_eV" in exp:
        lo, hi = exp["gap_eV"]
        checks.append(_check("band_gap", gap is not None and lo <= gap <= hi,
                             f"{gap:.3f} eV, expected {lo}-{hi}" if gap is not None else f"no gap, expected {lo}-{hi}"))
    soc = exp.get("soc", "optional")
    if soc in ("required", "forbidden"):
        checks.append(_check("spin_orbit", facts["soc"] == (soc == "required"),
                             f"spin-orbit {'on' if facts['soc'] else 'off'}, {soc}"))
    if exp.get("magnetic"):
        m = facts.get("magnetization_per_atom")
        lo, hi = exp["magnetic"]
        checks.append(_check("magnetization", m is not None and lo <= m <= hi,
                             f"{m} muB/atom, expected {lo}-{hi}" if m is not None else "not spin-polarized"))
    ref = REFERENCES / f"{name}.json"
    if ref.exists():
        checks.append(compare_to_reference(json.loads(ref.read_text()), band_fingerprint(work_dir, metal)))
    plot_script = _plot_script(work_dir)
    checks.append(_check("plot_uses_kpath_tool", bool(plot_script and "plot_axis" in plot_script),
                         "plot script reads chem_llm.kpath.plot_axis" if plot_script and "plot_axis" in plot_script
                         else "plot script does not use plot_axis (risk: wrong ticks or lines across breaks)",
                         hard=False))
    hard = [c for c in checks if c["hard"]]
    return {"case": name, "passed": all(c["ok"] for c in hard), "checks": checks,
            "facts": {k: (str(v) if isinstance(v, Path) else v) for k, v in facts.items()}}


def _plot_script(work_dir: Path) -> str | None:
    for path in sorted(Path(work_dir).glob("*.py")):
        text = _read(path) or ""
        if "savefig" in text:
            return text
    return None


def compare_to_reference(ref: dict, fp: dict | None, tol: float | None = None) -> dict:
    """Golden high-symmetry-point check."""
    tol = tol or ref.get("tolerance_eV", 0.15)
    if fp is None:
        return _check("reference_bands", False, "no bands output on a kpath.json path to compare")
    if fp["soc"] != ref["soc"]:
        return _check("reference_bands", False, f"spin-orbit {'on' if fp['soc'] else 'off'}, reference has it "
                                                f"{'on' if ref['soc'] else 'off'}")
    missing = sorted(set(ref["points"]) - set(fp["points"]))
    if missing:
        return _check("reference_bands", False, f"high-symmetry points missing from the path: {missing}")
    worst, where = 0.0, ""
    for label, ref_e in ref["points"].items():
        got = fp["points"][label]
        n = min(len(got), len(ref_e))
        for i in range(n):
            d = abs(got[i] - ref_e[i])
            if d > worst:
                worst, where = d, f"{label}, band {fp['band_window'][0] + i:+d}"
    return _check("reference_bands", worst <= tol,
                  f"max |dE| {worst:.3f} eV at {where or '-'} over {len(ref['points'])} points (tolerance {tol})")


def promote(run_dir: Path, tolerance: float = 0.15) -> Path:
    """Make a validated run the golden reference for its case."""
    from .archive import read_index

    run_dir = Path(run_dir).resolve()
    record = json.loads((run_dir / "record.json").read_text())
    rel = str(run_dir.relative_to(BENCH))
    name = next((e["case"] for e in read_index(BENCH) if e.get("dir") == rel), None) or record.get("case")
    if name is None:
        raise SystemExit(f"{run_dir}: not in bench/index.jsonl, so its case is unknown")
    _, cases = load_suites()
    case = cases[name]
    fp = band_fingerprint(Path(record["work_dir"]), bool((case.get("expect") or {}).get("metal")))
    if fp is None:
        raise SystemExit(f"{run_dir}: no bands on a kpath.json path in {record['work_dir']} (was it overwritten?)")
    fp.update({"case": name, "tolerance_eV": tolerance, "from_run": str(run_dir.relative_to(BENCH)),
               "promoted": datetime.now().isoformat(timespec="seconds"),
               "band_gap_eV": (record.get("outcome") or {}).get("band_gap")})
    REFERENCES.mkdir(parents=True, exist_ok=True)
    out = REFERENCES / f"{name}.json"
    out.write_text(json.dumps(fp, indent=1))
    return out


# ----------------------------------------------------------------------
# running
def run_case(name: str, *, suite: str | None, label: str | None, backend: str | None,
             max_steps: int | None = None) -> dict:
    """Run one case in this process (called in a subprocess by `run`)."""
    import functools

    from .agent import DFTAgent
    from .settings import Settings

    _, cases = load_suites()
    case = cases[name]
    settings = Settings.discover(REPO_ROOT)
    if settings.archive_dir is None:
        settings.archive_dir = BENCH
    agent = DFTAgent(settings, backend=backend, max_steps=max_steps)
    result = agent.run(
        case["compound"], case["spacegroup"],
        spacegroup_number=case.get("spacegroup_number"), spacegroup_name=case.get("name", ""),
        work_dir=work_dir_of(name, case), clear_dir=True, verbose=False,
        case=name, label=label, meta={"suite": suite, "purpose": case.get("purpose")},
        evaluate=lambda wd, rec: evaluate_case(name, case, wd, rec),
    )
    ev = result.evaluation or {}
    return {"case": name, "passed": ev.get("passed"), "failed": [c["name"] for c in ev.get("checks", [])
                                                                 if c.get("ok") is False and c.get("hard")],
            "gap": result.band_gap, "steps": result.num_steps, "wall_min": round(result.runtime_seconds / 60, 1),
            "archive": str(result.archive_dir), "outcome": {"checks": ev.get("checks", [])}}


def _subprocess_case(name: str, suite: str | None, label: str | None, backend: str | None,
                     max_steps: int | None) -> dict:
    logs = BENCH / "work" / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "chem_llm.bench", "_run-case", name]
    for flag, value in (("--suite", suite), ("--label", label), ("--backend", backend),
                        ("--max-steps", max_steps)):
        if value is not None:
            cmd += [flag, str(value)]
    started = time.time()
    with open(logs / f"{name}.log", "w") as log:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=log, text=True, cwd=REPO_ROOT)
    last = (proc.stdout.strip().splitlines() or [""])[-1]
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return {"case": name, "passed": False, "error": f"exit {proc.returncode}; see {logs / (name + '.log')}",
                "wall_min": round((time.time() - started) / 60, 1)}


def run_many(names: list[str], *, suite: str | None, label: str | None, backend: str | None,
             parallel: int = 1, max_steps: int | None = None) -> list[dict]:
    results = []
    with futures.ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        jobs = {pool.submit(_subprocess_case, n, suite, label, backend, max_steps): n for n in names}
        for job in futures.as_completed(jobs):
            r = job.result()
            results.append(r)
            status = "PASS" if r.get("passed") else "FAIL"
            gap = f"{r['gap']:.3f} eV" if r.get("gap") is not None else "no gap"
            print(f"[{datetime.now():%H:%M}] {status} {r['case']:<10} {gap:>10}  {r.get('steps', '-')} steps  "
                  f"{r.get('wall_min', '-')} min  {', '.join(r.get('failed') or []) or r.get('error', '')}", flush=True)
    write_report()
    return results


# ----------------------------------------------------------------------
# LangSmith
def langsmith_examples(names: list[str]) -> list[dict]:
    _, cases = load_suites()
    return [{"inputs": {"case": n, **{k: cases[n][k] for k in ("compound", "spacegroup") if k in cases[n]}},
             "outputs": {"expect": cases[n].get("expect") or {}, "purpose": cases[n].get("purpose")}}
            for n in names]


def langsmith_scores(outputs: dict) -> dict:
    """LangSmith evaluator: every benchmark check as its own feedback key,
    plus passed / steps / wall time."""
    results = [{"key": "passed", "score": int(bool(outputs.get("passed")))}]
    for c in (outputs.get("outcome") or {}).get("checks", []):
        if c.get("ok") is not None:
            results.append({"key": c["name"], "score": int(bool(c["ok"])), "comment": c.get("detail")})
    for key in ("steps", "wall_min"):
        if outputs.get(key) is not None:
            results.append({"key": key, "score": outputs[key]})
    return {"results": results}


def run_langsmith(names: list[str], *, suite: str, label: str | None, backend: str | None, parallel: int,
                  max_steps: int | None = None):
    if not os.environ.get("LANGSMITH_API_KEY"):
        raise SystemExit("chem-llm-bench: --langsmith needs LANGSMITH_API_KEY (and LANGSMITH_TRACING=true to "
                         "trace the agent runs themselves); add them to .env or the environment")
    from langsmith import Client, evaluate

    from .agent_core import SYSTEM_PROMPT_TEMPLATE
    from .archive import code_version
    from .rundiff import _sha

    client = Client()
    dataset = f"chem_llm-{suite}"
    if not client.has_dataset(dataset_name=dataset):
        ds = client.create_dataset(dataset, description=f"chem_llm benchmark suite {suite!r} (bench/suites.yaml)")
        client.create_examples(dataset_id=ds.id, examples=langsmith_examples(names))
    os.environ.setdefault("LANGSMITH_PROJECT", f"chem_llm-{suite}")

    def target(inputs: dict) -> dict:
        return _subprocess_case(inputs["case"], suite, label, backend, max_steps)

    def scores(outputs: dict) -> dict:
        return langsmith_scores(outputs)

    code = code_version()
    results = evaluate(
        target, data=dataset, evaluators=[scores], max_concurrency=parallel,
        experiment_prefix=(label or "run").replace(" ", "-")[:40],
        metadata={"commit": code.get("commit"), "uncommitted_changes": bool(code.get("uncommitted_changes")), "backend": backend,
                  "system_prompt_sha": _sha(SYSTEM_PROMPT_TEMPLATE)},
    )
    write_report()
    return results


# ----------------------------------------------------------------------
# report
def write_report(path: Path = BENCH / "RESULTS.md") -> Path:
    from .archive import read_index

    suites, cases = load_suites()
    index = [e for e in read_index(BENCH) if e.get("case") in cases]
    by_case: dict[str, list[dict]] = {}
    for e in index:
        by_case.setdefault(e["case"], []).append(e)

    def mark(e):
        return {True: "pass", False: "**FAIL**"}.get(e.get("passed"), "-")

    def gap(e):
        return f"{e['band_gap']:.3f}" if isinstance(e.get("band_gap"), (int, float)) else "-"

    def expected(c):
        exp = c.get("expect") or {}
        return "metal" if exp.get("metal") else ("{}-{}".format(*exp["gap_eV"]) if "gap_eV" in exp else "-")

    def link(e, name):
        return f"[{name}](runs/{Path(e['dir']).name}/{name}.txt)" if e.get("dir") else "-"

    lines = ["# Benchmark results", "",
             f"Generated {datetime.now():%Y-%m-%d %H:%M} by `chem-llm-bench report` from `bench/index.jsonl`; do not "
             "edit by hand. Each run's folder under `bench/runs/` has its live progress, summary, the checks and "
             "what changed since the previous run of that compound.", ""]
    for sname, suite in suites.items():
        lines += [f"## Suite `{sname}`", "", suite.get("description", ""), "",
                  "| case | what it tests | last run | result | failed checks | gap eV (expected) | steps | wall min | prompt | commit | records |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name in suite["cases"]:
            c, runs = cases[name], by_case.get(name, [])
            if not runs:
                lines.append(f"| {name} | {c.get('purpose', '')} | never | - | - | - ({expected(c)}) | - | - | - | - | - |")
                continue
            e = runs[-1]
            lines.append(
                f"| {name} | {c.get('purpose', '')} | {(e.get('time') or '')[:16]} | {mark(e)} | "
                f"{', '.join(e.get('failed_checks') or []) or '-'} | {gap(e)} ({expected(c)}) | {e.get('steps', '-')} | "
                f"{e.get('wall_min', '-')} | {(e.get('prompt_version') or '-')[:8]} | "
                f"{((e.get('code') or {}).get('commit') or '-')[:8]} | "
                f"{link(e, 'summary')}, {link(e, 'progress')}, {link(e, 'changes')} |")
        lines.append("")
    lines += ["## History", ""]
    for name, runs in by_case.items():
        lines += [f"### {name}: {cases[name].get('purpose', '')}", "",
                  "| date | label | backend | result | failed checks | gap eV | steps | wall min | model min | prompt | commit |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for e in reversed(runs[-25:]):
            lines.append(
                f"| {(e.get('time') or '')[:16]} | {e.get('label') or '-'} | {e.get('backend') or '-'} | {mark(e)} | "
                f"{', '.join(e.get('failed_checks') or []) or '-'} | {gap(e)} | {e.get('steps', '-')} | "
                f"{e.get('wall_min', '-')} | {e.get('llm_min') if e.get('llm_min') is not None else '-'} | "
                f"{(e.get('prompt_version') or '-')[:8]} | {((e.get('code') or {}).get('commit') or '-')[:8]} |")
        lines.append("")
    path.write_text("\n".join(lines))
    return path


# ----------------------------------------------------------------------
def import_history(entries: list[tuple[str, Path, Path | None]], label: str) -> None:
    """Record runs made before the benchmark existed (e.g. sandbox/test34)
    in the index, scored by today's checks, so the first new run has
    something to compare against. entries: (case, work_dir, progress_log)."""
    from .archive import RunArchive, commit_at
    from .report import load_log_entry, summarize_log

    _, cases = load_suites()
    for name, work_dir, progress in entries:
        record, _ = load_log_entry(work_dir)
        record["code"] = {"commit": commit_at(record["timestamp"]), "inferred": True}
        record.setdefault("config", {})
        run = RunArchive(BENCH, name, label, {"suite": "imported", "purpose": cases[name].get("purpose")})
        if progress and Path(progress).exists():
            run.progress_stream.write(Path(progress).read_text())
        ev = evaluate_case(name, cases[name], Path(work_dir), record)
        (run.dir / "checks.json").write_text(json.dumps(ev, indent=1, default=str))
        from .pipeline import _inspect_outputs
        found = _inspect_outputs(Path(work_dir))
        record["outcome"] = {"band_gap": found.get("band_gap"), "dft_finished": found.get("dft_finished")}
        run.finish(record, summarize_log(work_dir), {"passed": ev["passed"], "time": record["timestamp"],
                   "failed_checks": [c["name"] for c in ev["checks"] if c.get("ok") is False]})
        print(name, "PASS" if ev["passed"] else "FAIL", [c["name"] for c in ev["checks"] if not c["ok"]])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="chem-llm-bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="suites and cases")
    r = sub.add_parser("run", help="run suites and/or cases")
    r.add_argument("names", nargs="+")
    r.add_argument("--parallel", type=int, default=1, help="cases at once (vLLM backend)")
    r.add_argument("--label", default=None, help="why this sweep: shows in the history and changes.txt")
    r.add_argument("--backend", default="vllm", help="qwen | vllm (default) | claude")
    r.add_argument("--max-steps", type=int, default=None)
    r.add_argument("--langsmith", action="store_true", help="also record a LangSmith experiment")
    one = sub.add_parser("_run-case")
    one.add_argument("name")
    one.add_argument("--suite")
    one.add_argument("--label")
    one.add_argument("--backend")
    one.add_argument("--max-steps", type=int)
    sub.add_parser("report", help="regenerate bench/RESULTS.md")
    pr = sub.add_parser("promote", help="make a validated run its case's golden reference")
    pr.add_argument("run_dir")
    pr.add_argument("--tolerance", type=float, default=0.15)
    args = p.parse_args(argv)

    if args.cmd == "list":
        suites, cases = load_suites()
        for sname, s in suites.items():
            print(f"{sname}: {s.get('description', '')}")
            for n in s["cases"]:
                c = cases[n]
                print(f"    {n:<10} {c['compound']:<9} {c['spacegroup']:<10} {c.get('purpose', '')}")
        return 0
    if args.cmd == "_run-case":
        out = run_case(args.name, suite=args.suite, label=args.label, backend=args.backend, max_steps=args.max_steps)
        print(json.dumps(out, default=str))
        return 0
    if args.cmd == "run":
        suites, _ = load_suites()
        suite = args.names[0] if len(args.names) == 1 and args.names[0] in suites else "adhoc"
        names = resolve(args.names)
        print(f"running {len(names)} case(s), {args.parallel} at a time: {' '.join(names)}", flush=True)
        if args.langsmith:
            run_langsmith(names, suite=suite, label=args.label, backend=args.backend, parallel=args.parallel,
                          max_steps=args.max_steps)
        else:
            results = run_many(names, suite=suite, label=args.label, backend=args.backend, parallel=args.parallel,
                               max_steps=args.max_steps)
            print(f"{sum(bool(r.get('passed')) for r in results)}/{len(results)} passed; see bench/RESULTS.md")
        return 0
    if args.cmd == "report":
        print(write_report())
        return 0
    if args.cmd == "promote":
        print(promote(Path(args.run_dir), args.tolerance))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
