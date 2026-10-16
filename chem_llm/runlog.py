"""The run record: one schema for log.jsonl (work dir) and record.json (archive).

Format version 2 (2026-10-16), field by field:

    format_version     2
    run_id             links the record to its trace.jsonl lines and the index
    timestamp          when the run finished
    compound, spacegroup, spacegroup_number (only when given)
    model, work_dir, completed
    total_steps        agent steps (one tool call each)
    runtime            wall-clock seconds
    total_tokens       prompt + completion tokens over the whole run
    outcome            what the run left on disk: band gap, convergence, plot
    prompt             {"system": raw system prompt, "task": raw task text}
    steps              [{step, phase, tool, args, result, llm_time, tool_time,
                         tokens}] -- every tool call in order, with its full
                       arguments and result; "phase" is the workflow STEP it
                       belongs to (chem_llm.telemetry.assign_phases)
    config             the run's settings: backend, model, temperature,
                       do_sample, max_new_tokens, step/history limits, QE
                       launch, tools, code_commit, uncommitted_changes
    metrics            {operations, phases, n_tools, qe_stages}
                       (chem_llm.telemetry.run_metrics)

log.jsonl keeps tool results whole; record.json cuts strings longer than
RESULT_CAP characters (saying so). Format 1 records -- everything logged
before 2026-10-16, with a "final_state" -- are read through upgrade(), so
replay, --summarize-log, --diff-runs and the benchmark accept both.
"""
from __future__ import annotations

from types import SimpleNamespace

FORMAT_VERSION = 2
RESULT_CAP = 20_000  # characters per string inside a tool result, in record.json


def cap(value):
    """Cut strings longer than RESULT_CAP, recursively, saying how much was cut."""
    if isinstance(value, str) and len(value) > RESULT_CAP:
        return value[:RESULT_CAP] + f" ... [{len(value) - RESULT_CAP:,} more characters cut in the archive]"
    if isinstance(value, dict):
        return {k: cap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cap(v) for v in value]
    return value


def flat_config(config: dict | None) -> dict:
    """Format-1 configs nested the sampling settings under "decoding"."""
    cfg = dict(config or {})
    decoding = cfg.pop("decoding", None)
    if decoding is not None:
        cfg.setdefault("temperature", decoding.get("temperature", 0.0 if decoding.get("do_sample") is False else None))
        for key in ("do_sample", "max_new_tokens", "max_tokens", "effort"):
            if key in decoding:
                cfg.setdefault(key, decoding[key])
    for k in ("system_prompt_chars", "task_chars"):
        cfg.pop(k, None)
    return cfg


def _steps(history: list[dict], timing: list[dict]) -> list[dict]:
    from .telemetry import PHASES, assign_phases

    if not timing:  # runs logged before per-step timing existed
        return [{"step": e.get("step"), "tool": e.get("tool"), "args": e.get("args"), "result": e.get("result")}
                for e in history]
    by_index = {e.get("step"): e for e in history}
    out = []
    for rec, phase in zip(timing, assign_phases(history, timing)):
        entries = [by_index[i] for i in rec.get("history", []) if i in by_index]
        main = entries[-1] if entries else {}
        step = {"step": rec.get("step"), "phase": PHASES[phase - 1], "tool": main.get("tool", rec.get("tool")),
                "args": main.get("args"), "result": main.get("result"),
                "llm_time": rec.get("llm_s"), "tool_time": rec.get("tool_s"),
                "tokens": (rec.get("prompt_tokens") or 0) + (rec.get("completion_tokens") or 0)
                if rec.get("prompt_tokens") is not None else None}
        if len(entries) > 1:
            step["also_logged"] = [{"tool": e["tool"], "args": e.get("args"), "result": e.get("result")}
                                   for e in entries[:-1]]
        out.append(step)
    return out


def build_record(*, run_id, timestamp, history, timing, task, system_prompt, completed, runtime, model,
                 work_dir, config, metrics=None, outcome=None, compound=None, spacegroup=None,
                 spacegroup_number=None, model_load_s=None) -> dict:
    """A format-2 record from a run's raw history and per-step timing."""
    from .telemetry import run_metrics

    if not metrics or "operations" not in metrics:
        metrics = run_metrics(SimpleNamespace(history=history, steps=timing, task=task), runtime or 0, model_load_s) \
            if timing else {}
    metrics = {k: metrics[k] for k in ("operations", "phases", "n_tools", "qe_stages") if k in metrics}
    cif = next((e.get("args") or {} for e in history if e.get("tool") == "generate_cif"), {})
    record = {"format_version": FORMAT_VERSION, "run_id": run_id, "timestamp": timestamp,
              "compound": compound or cif.get("composition"),
              "spacegroup": spacegroup or cif.get("spacegroup_symbol")}
    number = spacegroup_number or cif.get("spacegroup_number")
    if number:
        record["spacegroup_number"] = number
    record.update({
        "model": model,
        "work_dir": str(work_dir) if work_dir else None,
        "completed": completed,
        "total_steps": len(history),
        "runtime": round(runtime or 0, 1),
        "total_tokens": (metrics.get("operations") or {}).get("total", {}).get("tokens"),
        "outcome": outcome,
        "prompt": {"system": system_prompt, "task": task},
        "steps": _steps(history, timing),
        "config": flat_config(config),
        "metrics": metrics,
    })
    return record


def upgrade(entry: dict) -> dict:
    """Any log record -> format 2 (format-2 records are returned as they are)."""
    if entry.get("format_version") == FORMAT_VERSION:
        return entry
    fs = entry.get("final_state") or {}
    config = dict(entry.get("config") or {})
    code = entry.get("code") or {}
    if code.get("commit"):
        from .archive import _is_code

        config.setdefault("code_commit", code["commit"])
        config.setdefault("uncommitted_changes",
                          code.get("uncommitted_changes", [f for f in code.get("dirty") or [] if _is_code(f)]))
    meta = entry.get("meta") or {}
    rec = build_record(
        run_id=entry.get("run_id"), timestamp=entry.get("timestamp"),
        history=fs.get("history") or [], timing=fs.get("steps") or [], task=fs.get("task"),
        system_prompt=(entry.get("prompt") or {}).get("system"), completed=entry.get("completed"),
        runtime=entry.get("runtime"), model=entry.get("model"), work_dir=entry.get("work_dir"),
        config=config, metrics=entry.get("metrics"), outcome=entry.get("outcome"),
        compound=meta.get("compound"), spacegroup=meta.get("spacegroup"),
        spacegroup_number=meta.get("spacegroup_number"),
        model_load_s=(entry.get("metrics") or {}).get("model_load_s") or entry.get("model_load_s"),
    )
    if entry.get("replayed_from"):
        rec["replayed_from"] = entry["replayed_from"]
    return rec


def history_of(record: dict) -> list[dict]:
    """The run's tool calls as [{step, tool, args, result}] in order (what
    the summary, --diff-runs and replay read), from a record of any format."""
    record = upgrade(record)
    out = []
    for s in record.get("steps") or []:
        for e in s.get("also_logged") or []:
            out.append({"step": len(out) + 1, "tool": e["tool"], "args": e.get("args"), "result": e.get("result")})
        out.append({"step": len(out) + 1, "tool": s["tool"], "args": s.get("args"), "result": s.get("result")})
    return out


def final_result_of(record: dict) -> str | None:
    """The agent's done summary, if it called done."""
    done = [e for e in history_of(record) if e["tool"] == "done"]
    return (done[-1].get("args") or {}).get("summary") if done else None


def for_archive(record: dict) -> dict:
    """record.json: the format-2 record with long tool results cut."""
    out = dict(upgrade(record))
    out["steps"] = [{**s, "result": cap(s.get("result")),
                     **({"also_logged": cap(s["also_logged"])} if s.get("also_logged") else {})}
                    for s in out.get("steps") or []]
    return out
