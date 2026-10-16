"""Where a run's time and tokens go.

Each agent step records how long the model took to produce its tool call,
how long the tool took, and the prompt/completion token counts (when the
backend reports them):

    AgentState.steps   one record per step, saved in the run's log.jsonl
    trace.jsonl        the same records plus the step's history entries,
                       appended as each step finishes -- so a run killed
                       mid-way (an expired SLURM allocation) still leaves
                       its trajectory and timings on disk
    run_metrics()      the run-level roll-up stored as log.jsonl "metrics"
                       (and in the archived record.json):
        operations   time and tokens per kind of work: llm (the model
                     generating), tools (every tool except the two below:
                     Materials Project, pseudopotentials, doc search, files,
                     notes, k-path, validation), qe (run_espresso_workflow),
                     code (the agent's scripts, run_python), and total
        phases       time and tokens per workflow STEP of the task (1-7)
        n_tools      how many times each tool was called
        qe_stages    seconds per Quantum ESPRESSO stage

format_timing() renders the roll-up for the run summary.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

# Operations: what kind of work a step's tool call is.
OPERATION_OF_TOOL = {"run_espresso_workflow": "qe", "run_python": "code"}   # everything else: "tools"
OPERATIONS = ("llm", "tools", "qe", "code", "total")
OPERATION_LABELS = {"llm": "model generation", "tools": "tool calls (databases, docs, files)",
                    "qe": "Quantum ESPRESSO", "code": "running the agent's scripts", "total": "total"}

# Phases: the task's workflow STEPs (chem_llm.tasks).
PHASES = ("1_structure", "2_perturbed_structures", "3_pseudopotentials", "4_qe_settings",
          "5_qe_inputs", "6_run_qe", "7_plot")
PHASE_LABELS = {"1_structure": "1 fetch + reduce structure", "2_perturbed_structures": "2 perturbed structures",
                "3_pseudopotentials": "3 pseudopotentials", "4_qe_settings": "4 decide QE settings",
                "5_qe_inputs": "5 write + validate QE inputs", "6_run_qe": "6 run QE", "7_plot": "7 plot + done"}

class StepRecorder:
    """Times the agent loop's steps. The loop calls begin() before asking
    the model, llm_done() when it has a tool call, tool_done() after
    running it, and end() to close the step."""

    def __init__(self, state, trace_file: Path | None = None, run_id: str | None = None):
        self.state = state
        self.trace_file = Path(trace_file) if trace_file else None
        self.run_id = run_id
        self.t0 = time.perf_counter()
        self._rec: dict | None = None
        self._before = 0

    def begin(self, step: int) -> None:
        self._before = len(self.state.history)
        self.state.llm_usage = None
        self._rec = {"step": step, "t_start": round(time.perf_counter() - self.t0, 3),
                     "tool": None, "llm_s": None, "tool_s": None}

    def llm_done(self, seconds: float) -> None:
        self._rec["llm_s"] = round(seconds, 3)
        usage = self.state.llm_usage or {}
        for key in ("prompt_tokens", "completion_tokens", "prompt_chars", "cached_tokens"):
            if usage.get(key) is not None:
                self._rec[key] = usage[key]

    def tool_done(self, tool: str, seconds: float) -> None:
        self._rec["tool"] = tool
        self._rec["tool_s"] = round(seconds, 3)

    def end(self) -> None:
        rec, self._rec = self._rec, None
        new = self.state.history[self._before:]
        if rec["tool"] is None and new:
            rec["tool"] = new[-1]["tool"]        # parse_error / api_error / replayed entry
        if new and new[-1]["tool"] == "refused":
            rec["refused"] = True
        rec["t_end"] = round(time.perf_counter() - self.t0, 3)
        rec["history"] = [e["step"] for e in new]
        self.state.steps.append(rec)
        if self.trace_file is not None:
            # the same step fields as a format-2 record's steps (chem_llm.runlog)
            main = new[-1] if new else {}
            line = {"format_version": 2, "run_id": self.run_id, "step": rec["step"], "tool": rec["tool"],
                    "args": main.get("args"), "result": main.get("result"),
                    "llm_time": rec["llm_s"], "tool_time": rec["tool_s"],
                    "tokens": (rec.get("prompt_tokens") or 0) + (rec.get("completion_tokens") or 0)
                    if rec.get("prompt_tokens") is not None else None,
                    "t_start": rec["t_start"], "t_end": rec["t_end"]}
            if len(new) > 1:
                line["also_logged"] = [{"tool": e["tool"], "args": e.get("args"), "result": e.get("result")}
                                       for e in new[:-1]]
            with self.trace_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, default=str) + "\n")


_FILE_PHASE = (  # (substring of a path, phase), first match wins
    ("plot", 7), ("bands.dat", 7), (".gnu", 7), (".pdf", 7),
    ("setup_jobs", 5), ("pw.in", 5), ("bands.in", 5), ("bands_post.in", 5), ("kpath", 5),
    (".out", 6),
    ("generate_structures", 2), ("structure_0", 2),
    ("reduce", 1), ("primitive", 1), ("base", 1), ("spglib", 1), ("check", 1), ("probe", 1),
    ("template/", 3), (".upf", 3), (".UPF", 3),
)


def _phase_of_entry(e: dict) -> int | None:
    """The workflow STEP a history entry belongs to, from what it did; None
    for entries that say nothing about it (doc searches, refusals, ...)."""
    tool, args = e.get("tool"), e.get("args") or {}
    fixed = {"generate_cif": 1, "get_pseudopotential": 3, "get_kpath": 5, "validate_qe_input": 5,
             "run_espresso_workflow": 6, "done": 7}
    if tool in fixed:
        return fixed[tool]
    if tool == "note":
        m = re.search(r"\bSTEP\s*(\d)", str(args.get("text", "")))
        return int(m.group(1)) if m and 1 <= int(m.group(1)) <= 7 else None
    path = str(args.get("path") or args.get("directory") or "")
    for key, phase in _FILE_PHASE:
        if key in path:
            return phase
    return None


def assign_phases(history: list[dict], steps: list[dict]) -> list[int]:
    """One workflow STEP (1-7) per agent step. A step's own action decides
    it (a generate_cif call is STEP 1, writing setup_jobs.py is STEP 5, a
    note that says "STEP 4" is STEP 4); steps that say nothing about it --
    a doc search, a read of an unfamiliar file -- belong to the step they
    prepare, i.e. the next one that does say, else the previous one."""
    by_index = {e.get("step"): e for e in history}
    found: list[int | None] = []
    for rec in steps:
        entries = [by_index[i] for i in rec.get("history", []) if i in by_index]
        found.append(next((p for p in (_phase_of_entry(e) for e in reversed(entries)) if p), None))
    out, upcoming = [0] * len(found), None
    for i in range(len(found) - 1, -1, -1):
        upcoming = found[i] or upcoming
        out[i] = upcoming or 0
    last = 1
    for i, p in enumerate(out):
        out[i] = p or last
        last = out[i]
    return out


def _qe_stage_times(history: list[dict]) -> dict:
    """Seconds per QE stage (scf / bands_scf / bands_post), summed over every
    run_espresso_workflow call, from the timings the tool reports."""
    out: dict[str, float] = {}
    for e in history:
        r = e.get("result")
        if e["tool"] == "run_espresso_workflow" and isinstance(r, dict):
            for stage, s in (r.get("timings_s") or {}).items():
                out[stage] = round(out.get(stage, 0.0) + s, 1)
    return out


def _tokens(rec: dict) -> int:
    return (rec.get("prompt_tokens") or 0) + (rec.get("completion_tokens") or 0)


def run_metrics(state, wall_s: float, model_load_s: float | None = None) -> dict:
    """The run-level roll-up stored with the run's log record.

    operations: {llm, tools, qe, code, total} -> {"time": s, "tokens": n}.
    Every token is spent by the model, so llm.tokens is the run's total;
    the tokens of tools / qe / code are those of the model calls that chose
    those calls (what deciding them cost). total.time is the run's wall time.
    phases: the same {"time", "tokens"} per workflow STEP (assign_phases);
    a step's time is its model time plus its tool time.
    """
    steps = getattr(state, "steps", []) or []
    history = state.history
    ops = {k: {"time": 0.0, "tokens": 0} for k in OPERATIONS}
    n_tools: dict[str, int] = {}
    for rec in steps:
        ops["llm"]["time"] += rec.get("llm_s") or 0
        ops["llm"]["tokens"] += _tokens(rec)
        if rec.get("tool_s") is None:
            continue
        op = OPERATION_OF_TOOL.get(rec["tool"], "tools")
        ops[op]["time"] += rec["tool_s"]
        ops[op]["tokens"] += _tokens(rec)
        n_tools[rec["tool"]] = n_tools.get(rec["tool"], 0) + 1
    ops["total"] = {"time": wall_s, "tokens": ops["llm"]["tokens"]}

    phases = {k: {"time": 0.0, "tokens": 0} for k in PHASES}
    for rec, phase in zip(steps, assign_phases(history, steps)):
        key = PHASES[phase - 1]
        phases[key]["time"] += (rec.get("llm_s") or 0) + (rec.get("tool_s") or 0)
        phases[key]["tokens"] += _tokens(rec)

    for d in (*ops.values(), *phases.values()):
        d["time"] = round(d["time"], 1)
    out = {"operations": ops, "phases": phases,
           "n_tools": dict(sorted(n_tools.items(), key=lambda kv: -kv[1])),
           "qe_stages": _qe_stage_times(history)}
    if model_load_s is not None:
        out["model_load_s"] = round(model_load_s, 1)
    return out


def operation_times(metrics: dict | None) -> dict:
    """{llm, tools, qe, code, total} seconds from a metrics dict of either
    shape (runs logged before 2026-10-16 used phases_s)."""
    m = metrics or {}
    if "operations" in m:
        return {k: v["time"] for k, v in m["operations"].items()}
    p = m.get("phases_s") or {}
    if not p:
        return {}
    qe = p.get("qe", p.get("quantum_espresso", 0.0))
    code = p.get("run_code", p.get("agent_scripts", 0.0))
    tools = sum(v for k, v in p.items() if k not in ("llm", "qe", "quantum_espresso", "run_code",
                                                      "agent_scripts", "harness"))
    return {"llm": p.get("llm", 0.0), "tools": tools, "qe": qe, "code": code, "total": m.get("wall_s")}


def _mmss(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def format_timing(metrics: dict | None) -> list[str]:
    """Readable lines for the run summary's timing section."""
    if not metrics:
        return ["    not recorded (this run predates timing logs)"]
    if "operations" not in metrics:
        times = operation_times(metrics)
        wall = times.get("total") or 0
        return [f"    wall time {_mmss(wall)}"] + [
            f"      {OPERATION_LABELS[k]:<36} {_mmss(times[k]):>7}" for k in ("llm", "tools", "qe", "code") if k in times]
    ops = metrics["operations"]
    wall, total_tokens = ops["total"]["time"] or 0, ops["total"]["tokens"]
    lines = [f"    wall time {_mmss(wall)}, {total_tokens:,} tokens"
             + (f" (+ {_mmss(metrics['model_load_s'])} loading the model before the run)"
                if (metrics.get("model_load_s") or 0) >= 1 else "")]
    lines.append(f"      {'by operation':<36} {'time':>7} {'share':>6} {'tokens':>11}")
    for k in ("llm", "tools", "qe", "code"):
        d = ops[k]
        share = f"{100 * d['time'] / wall:5.0f}%" if wall else ""
        lines.append(f"      {OPERATION_LABELS[k]:<36} {_mmss(d['time']):>7} {share:>6} {d['tokens']:>11,}")
    lines.append(f"      {'by workflow step':<36} {'time':>7} {'share':>6} {'tokens':>11}")
    for k, d in metrics.get("phases", {}).items():
        share = f"{100 * d['time'] / wall:5.0f}%" if wall else ""
        lines.append(f"      {PHASE_LABELS.get(k, k):<36} {_mmss(d['time']):>7} {share:>6} {d['tokens']:>11,}")
    qe = metrics.get("qe_stages") or {}
    if qe:
        lines.append("    Quantum ESPRESSO by stage: " + ", ".join(f"{k} {_mmss(v)}" for k, v in qe.items()))
    if metrics.get("n_tools"):
        lines.append("    tool calls: " + ", ".join(f"{k} {v}" for k, v in metrics["n_tools"].items()))
    return lines
