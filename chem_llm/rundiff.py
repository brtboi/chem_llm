"""Prompt versions, and side-by-side comparison of two recorded runs.

Every run's log record carries a "prompt" snapshot: the full system prompt
(which embeds the tool schemas) and short hashes of it and of the task
text. Runs with the same "version" saw the same instructions, so a change
in behaviour between them comes from sampling, the model or the settings.

    chem-llm --diff-runs sandbox/test34/zno@0 sandbox/test34/zno

compares two runs: outcome, steps, where the time went, tool usage, the
settings that differ, and a unified diff of the system prompt and task.
A run is named by a work dir or log.jsonl, optionally with @N to pick the
N-th record in that log (default @-1, the latest).
"""
from __future__ import annotations

import difflib
import hashlib
from collections import Counter
from pathlib import Path


def _sha(text: str | None) -> str | None:
    return hashlib.sha256(text.encode()).hexdigest()[:12] if text is not None else None


def prompt_snapshot(system_prompt: str, task: str) -> dict:
    """What a run's log records about its prompt. `version` changes when
    either the system prompt or the task text does."""
    return {
        "version": _sha(_sha(system_prompt) + _sha(task)),
        "system_sha": _sha(system_prompt),
        "task_sha": _sha(task),
        "system_chars": len(system_prompt),
        "task_chars": len(task),
        "system": system_prompt,
    }


def parse_run_spec(spec: str) -> tuple[Path, int]:
    """'dir', 'dir/log.jsonl', 'dir@0', 'dir@-2' -> (path, index)."""
    path, sep, index = spec.rpartition("@")
    if sep and index.lstrip("-").isdigit():
        return Path(path), int(index)
    return Path(spec), -1


def _load(spec: str) -> dict:
    from .report import load_log_entry

    from .runlog import upgrade

    path, index = parse_run_spec(spec)
    entry, work_dir = load_log_entry(path, index)
    entry = dict(upgrade(entry))
    entry["_label"] = f"{path}@{index}"
    entry["_work_dir"] = work_dir
    return entry


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _numbers(entry: dict) -> dict:
    """Comparable figures for one run; None where the run did not record them."""
    from .history import run_succeeded

    from .runlog import history_of

    history = history_of(entry)
    calls = Counter(e["tool"] for e in history)
    from .telemetry import operation_times

    m = entry.get("metrics") or {}
    times = operation_times(m)
    tokens = entry.get("total_tokens")
    outcome = entry.get("outcome") or {}
    minutes = lambda k: round(times[k] / 60, 1) if times.get(k) is not None else None
    return {
        "completed": entry.get("completed"),
        "band gap (eV)": outcome.get("band_gap"),
        "steps": len(history),
        "wall time (min)": round(entry["runtime"] / 60, 1) if entry.get("runtime") else None,
        "model generation (min)": minutes("llm"),
        "tool calls (min)": minutes("tools"),
        "Quantum ESPRESSO (min)": minutes("qe"),
        "agent scripts (min)": minutes("code"),
        "total tokens": tokens,
        "failed run_python": sum(1 for e in history if e["tool"] == "run_python" and not run_succeeded(e.get("result"))),
        "refused calls": calls.get("refused", 0),
        "unparseable replies": calls.get("parse_error", 0),
        "notes": calls.get("note", 0),
        "doc searches": calls.get("search_docs", 0),
    }


def _tool_counts(entry: dict) -> dict:
    m = entry.get("metrics") or {}
    if m.get("n_tools"):
        return dict(m["n_tools"])
    return {k: v["calls"] for k, v in (m.get("tools") or {}).items()}  # runs logged before 2026-10-16


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _table(rows: list[tuple[str, object, object]], a: str = "A", b: str = "B") -> list[str]:
    width = max(len(r[0]) for r in rows)
    out = [f"    {'':<{width}}  {a:>14}  {b:>14}"]
    for name, x, y in rows:
        mark = "" if _fmt(x) == _fmt(y) else "   *"
        out.append(f"    {name:<{width}}  {_fmt(x):>14}  {_fmt(y):>14}{mark}")
    return out


def _text_diff(a: str | None, b: str | None, name: str, context: int = 1, limit: int = 400) -> list[str]:
    if a is None or b is None:
        return [f"    {name}: not recorded for {'A' if a is None else 'B'} (run predates prompt logging)"]
    if a == b:
        return [f"    {name}: identical ({len(a):,} chars)"]
    diff = list(difflib.unified_diff(a.splitlines(), b.splitlines(), "A", "B", n=context, lineterm=""))
    out = [f"    {name}: {len(a):,} -> {len(b):,} chars"]
    out += ["      " + line for line in diff[2:limit + 2]]
    if len(diff) - 2 > limit:
        out.append(f"      ... {len(diff) - 2 - limit} more diff lines")
    return out


def diff_runs(spec_a: str, spec_b: str) -> str:
    """Readable comparison of two recorded runs (see the module docstring)."""
    a, b = _load(spec_a), _load(spec_b)
    lines = ["chem_llm run comparison", "=======================",
             f"A: {a['_label']}  ({a.get('timestamp', '?')[:19]}, {a.get('model', '?')})",
             f"B: {b['_label']}  ({b.get('timestamp', '?')[:19]}, {b.get('model', '?')})"]
    pa, pb = a.get("prompt") or {}, b.get("prompt") or {}
    va = prompt_snapshot(pa["system"], pa.get("task") or "")["version"] if pa.get("system") else None
    vb = prompt_snapshot(pb["system"], pb.get("task") or "")["version"] if pb.get("system") else None
    lines.append(f"prompt version: A {va or 'not recorded'}, B {vb or 'not recorded'}"
                 + ("  (same prompt)" if va and va == vb else ""))

    lines += ["", "Outcome and cost (* = differs)"]
    na, nb = _numbers(a), _numbers(b)
    lines += _table([(k, na[k], nb[k]) for k in na])

    ta, tb = _tool_counts(a), _tool_counts(b)
    if not ta:
        ta = Counter(s["tool"] for s in a.get("steps") or [])
    if not tb:
        tb = Counter(s["tool"] for s in b.get("steps") or [])
    lines += ["", "Tool calls"]
    lines += _table([(t, ta.get(t, 0), tb.get(t, 0)) for t in sorted(set(ta) | set(tb))])

    ca, cb = _flatten(a.get("config") or {}), _flatten(b.get("config") or {})
    changed = [(k, ca.get(k), cb.get(k)) for k in sorted(set(ca) | set(cb)) if ca.get(k) != cb.get(k)]
    lines += ["", "Settings that differ"]
    if not (a.get("config") and b.get("config")):
        lines.append("    settings not recorded for " + ("A" if not a.get("config") else "B"))
    elif not changed:
        lines.append("    none")
    for k, x, y in changed:
        lines.append(f"    {k}: {_fmt(x)} -> {_fmt(y)}")

    lines += ["", "Prompt"]
    lines += _text_diff(pa.get("system"), pb.get("system"), "system prompt")
    lines += _text_diff(pa.get("task"), pb.get("task"), "task")
    return "\n".join(lines) + "\n"
