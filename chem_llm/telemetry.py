"""Where a run's time and tokens go.

Each agent step records how long the model took to produce its tool call,
how long the tool took, and the prompt/completion token counts (when the
backend reports them):

    AgentState.steps   one record per step, saved in the run's log.jsonl
    trace.jsonl        the same records plus the step's history entries,
                       appended as each step finishes -- so a run killed
                       mid-way (an expired SLURM allocation) still leaves
                       its trajectory and timings on disk
    run_metrics()      the run-level roll-up stored as log.jsonl "metrics":
                       time per phase (model generation, Quantum ESPRESSO,
                       the agent's own scripts, doc search, ...), per tool,
                       per QE stage, per workflow STEP, and token totals

format_timing() renders the roll-up for the run summary.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

# Where a run's time goes, in four buckets: the model, the agent's own
# scripts (run_python), Quantum ESPRESSO (run_espresso_workflow), and every
# other tool call. What is left of the wall time (the harness itself) is
# well under 1% and not reported.
PHASE_OF_TOOL = {
    "run_espresso_workflow": "qe",
    "run_python": "run_code",
}
PHASE_LABELS = {
    "llm": "model generation",
    "run_code": "running the agent's scripts",
    "qe": "Quantum ESPRESSO",
    "other_tools": "other tool calls",
    # runs logged before 2026-10-16 used finer buckets
    "quantum_espresso": "Quantum ESPRESSO",
    "agent_scripts": "running the agent's scripts",
    "doc_search": "documentation search",
    "materials_project": "Materials Project fetch",
    "pseudopotentials": "pseudopotential fetch",
    "kpath_and_validation": "k-path / input validation",
    "harness": "harness overhead",
}


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
            line = {"run_id": self.run_id, **rec, "entries": new}
            with self.trace_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, default=str) + "\n")


def _workflow_stages(history: list[dict], steps: list[dict], task: str) -> list[dict]:
    """Time per workflow STEP, from the agent's own checkpoint notes ("STEP 3
    done: ..."): a stage ends at the first note that checkpoints it.
    Approximate by construction -- it is when the agent said so."""
    titles = {}
    for m in re.finditer(r"STEP (\d+):\s*([^.\n(]{0,60})", task or ""):
        titles.setdefault(int(m.group(1)), " ".join(m.group(2).split()[:6]))
    end_of_entry = {}
    for rec in steps:
        for h in rec.get("history", []):
            end_of_entry[h] = rec["t_end"]
    pattern = re.compile(r"\bSTEP\s*(\d+)\b[^.\n]{0,40}?\b(done|complete|completed|checkpoint|verified|finished)\b",
                         re.IGNORECASE)
    ends: dict[int, tuple[float, int]] = {}
    for e in history:
        if e["tool"] != "note":
            continue
        m = pattern.search(str((e.get("args") or {}).get("text", "")))
        if m and int(m.group(1)) not in ends and e["step"] in end_of_entry:
            ends[int(m.group(1))] = (end_of_entry[e["step"]], e["step"])
    stages, previous = [], 0.0
    for n in sorted(ends, key=lambda k: ends[k][0]):
        t, at = ends[n]
        stages.append({"stage": n, "title": titles.get(n, ""), "duration_s": round(t - previous, 1),
                       "ended_at_s": round(t, 1), "ended_at_step": at})
        previous = t
    return stages


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


def run_metrics(state, wall_s: float, model_load_s: float | None = None) -> dict:
    """The run-level roll-up stored with the run's log record."""
    steps = getattr(state, "steps", []) or []
    llm_s = sum(r.get("llm_s") or 0 for r in steps)
    tools: dict[str, dict] = {}
    for r in steps:
        if r.get("tool_s") is None:
            continue
        t = tools.setdefault(r["tool"], {"calls": 0, "seconds": 0.0})
        t["calls"] += 1
        t["seconds"] = round(t["seconds"] + r["tool_s"], 3)
    phases = {"llm": round(llm_s, 1), "run_code": 0.0, "qe": 0.0, "other_tools": 0.0}
    for tool, t in tools.items():
        phase = PHASE_OF_TOOL.get(tool, "other_tools")
        phases[phase] = round(phases[phase] + t["seconds"], 1)

    prompt = [r["prompt_tokens"] for r in steps if r.get("prompt_tokens") is not None]
    completion = [r["completion_tokens"] for r in steps if r.get("completion_tokens") is not None]
    gen_s = sum(r.get("llm_s") or 0 for r in steps if r.get("completion_tokens") is not None)
    llm = {"calls": sum(1 for r in steps if r.get("llm_s") is not None), "seconds": round(llm_s, 1)}
    if prompt:
        llm["prompt_tokens"] = {"total": sum(prompt), "mean": round(sum(prompt) / len(prompt)), "max": max(prompt)}
    if completion:
        llm["completion_tokens"] = {"total": sum(completion), "mean": round(sum(completion) / len(completion)),
                                    "max": max(completion)}
        if gen_s:
            llm["completion_tokens_per_s"] = round(sum(completion) / gen_s, 1)
    if llm["calls"]:
        llm["seconds_per_call"] = {"mean": round(llm_s / llm["calls"], 1),
                                   "max": round(max(r.get("llm_s") or 0 for r in steps), 1)}

    out = {"wall_s": round(wall_s, 1), "phases_s": phases, "llm": llm,
           "n_tools": {name: t["calls"] for name, t in sorted(tools.items(), key=lambda kv: -kv[1]["calls"])},
           "qe_stages_s": _qe_stage_times(state.history),
           "workflow_stages": _workflow_stages(state.history, steps, state.task)}
    if model_load_s is not None:
        out["model_load_s"] = round(model_load_s, 1)
    return out


def _mmss(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def format_timing(metrics: dict | None) -> list[str]:
    """Readable lines for the run summary's timing section."""
    if not metrics:
        return ["    not recorded (this run predates timing logs)"]
    wall = metrics.get("wall_s") or 0
    lines = [f"    wall time {_mmss(wall)}"
             + (f" (+ {_mmss(metrics['model_load_s'])} loading the model before the run)"
                if (metrics.get("model_load_s") or 0) >= 1 else "")]
    for phase, s in sorted(metrics.get("phases_s", {}).items(), key=lambda kv: -kv[1]):
        if phase not in ("llm", "qe", "quantum_espresso") and s < 0.05 * wall and s < 30:
            continue  # minor phases
        pct = f"{100 * s / wall:4.0f}%" if wall else ""
        lines.append(f"      {PHASE_LABELS.get(phase, phase):<28} {_mmss(s):>7} {pct}")
    llm = metrics.get("llm") or {}
    if llm.get("calls"):
        per = llm.get("seconds_per_call", {})
        text = f"    model: {llm['calls']} calls, {per.get('mean', 0):.0f} s each on average (max {per.get('max', 0):.0f} s)"
        if llm.get("prompt_tokens"):
            p = llm["prompt_tokens"]
            text += f"; prompt {p['mean']:,} tokens on average (max {p['max']:,})"
        if llm.get("completion_tokens"):
            c = llm["completion_tokens"]
            text += f"; replies {c['mean']:,} tokens on average"
        if llm.get("completion_tokens_per_s"):
            text += f"; {llm['completion_tokens_per_s']} tokens/s generated"
        lines.append(text)
    qe = metrics.get("qe_stages_s") or {}
    if qe:
        lines.append("    Quantum ESPRESSO by stage: " + ", ".join(f"{k} {_mmss(v)}" for k, v in qe.items()))
    stages = metrics.get("workflow_stages") or []
    if stages:
        lines.append("    by workflow step (ends at the agent's checkpoint note):")
        for st in stages:
            lines.append(f"      STEP {st['stage']} {st['title'][:40]:<40} {_mmss(st['duration_s']):>7}"
                         f"  (done at step {st['ended_at_step']})")
    return lines
