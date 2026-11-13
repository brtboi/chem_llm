"""What the model is shown of its own tool-call history.

The agent's full history is always kept (it is logged and replayable). What
goes back into the prompt is a *view* of its most recent entries. With
trimming on, that view drops material that is provably dead:

- note entries: every note is already in the prompt's Notes section;
- resolved fixes: once run_python(P) succeeds, the failed runs of P, the
  superseded versions of P, the reads of P and the doc searches made while P
  was the only failing script collapse into one line on the successful run
  that keeps the lesson ("errors fixed: ...");
- superseded outputs: a write of P that was later rewritten, or a read of P
  that was repeated (same range) or overtaken by a rewrite, keeps its call
  but loses its payload.

Nothing is dropped by age alone: the agent often reads several files and
then acts on all of them, so an output stays visible until it is superseded
or resolved -- or falls out of the max_history window, exactly as without
trimming. The window is applied first, so a trimmed view is never longer
than an untrimmed one.

progress_ledger() summarises the whole run (not just the window) so that
trimmed material is still accounted for: which scripts ran, which are
failing, what QE produced, which files were read.
"""
from __future__ import annotations

import os

__all__ = ["run_succeeded", "trimmed_view", "progress_ledger"]


def _path(entry: dict) -> str | None:
    path = (entry.get("args") or {}).get("path")
    return os.path.normpath(str(path)) if path else None


def run_succeeded(result) -> bool:
    """Whether a run_python result is a success: exit code 0. Results logged
    before run_python reported its exit code fall back to "no stderr"."""
    if not isinstance(result, dict):
        return False  # e.g. "ERROR executing run_python: ..."
    if "returncode" in result:
        return result["returncode"] == 0
    return not (result.get("stderr") or "").strip()


def _last_error_line(result) -> str:
    text = (result.get("stderr") or "") if isinstance(result, dict) else str(result)
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1][:160] if lines else "(no stderr)"


def _size(value) -> int:
    return len(value) if isinstance(value, str) else len(str(value))


def _read_range(entry: dict) -> tuple:
    args = entry.get("args") or {}
    return (args.get("start_line", 1), args.get("num_lines"))


def _decisions(history: list[dict]) -> tuple[set[int], dict[int, str], dict[int, str]]:
    """Scan the full history. Returns (indices to drop, index -> lesson line
    for a run that resolved a fix, index -> stub replacing a superseded
    payload)."""
    drop: set[int] = set()
    lessons: dict[int, str] = {}
    stubs: dict[int, str] = {}

    episode_start: dict[str, int] = {}   # path -> first write/run since its last success
    failing: set[str] = set()            # paths whose current episode has a failed run
    sole_failing_at: dict[int, str] = {}  # search index -> the only failing path then

    for i, e in enumerate(history):
        tool, path = e.get("tool"), _path(e)

        if tool == "note":
            drop.add(i)
        elif tool == "search_docs" and len(failing) == 1:
            sole_failing_at[i] = next(iter(failing))

        if tool in ("write_file", "run_python") and path:
            episode_start.setdefault(path, i)

        if tool == "run_python" and path:
            if not run_succeeded(e.get("result")):
                failing.add(path)
                continue
            start = episode_start.pop(path, i)
            failing.discard(path)
            episode = range(start, i)
            fails = [j for j in episode if history[j].get("tool") == "run_python" and _path(history[j]) == path]
            if not fails:
                continue  # ran first time: nothing to collapse
            writes = [j for j in episode if history[j].get("tool") == "write_file" and _path(history[j]) == path]
            drop.update(fails)
            drop.update(writes[:-1])
            drop.update(j for j in episode if history[j].get("tool") == "read_file" and _path(history[j]) == path)
            drop.update(j for j in episode if sole_failing_at.get(j) == path)
            errors = list(dict.fromkeys(_last_error_line(history[j].get("result")) for j in fails))
            lessons[i] = (
                f"{path} ran successfully after {len(fails)} failed attempt(s) "
                f"(steps {history[fails[0]].get('step')}-{history[fails[-1]].get('step')}, since removed "
                f"from this view); errors fixed: {errors}"
            )

    # Superseded payloads: a write overtaken by a later write of the same path;
    # a read overtaken by a later write, or by a later read of the same range.
    last_write: dict[str, int] = {}
    last_read: dict[tuple, int] = {}
    for i in range(len(history) - 1, -1, -1):
        e = history[i]
        tool, path = e.get("tool"), _path(e)
        if not path:
            continue
        if tool == "write_file":
            later = last_write.get(path)
            if later is not None and i not in drop:
                n = _size((e.get("args") or {}).get("content", ""))
                stubs[i] = f"[{n:,} chars -- superseded by the rewrite at step {history[later].get('step')}]"
            last_write[path] = i
        elif tool == "read_file":
            later_write = last_write.get(path)
            later_read = last_read.get((path, _read_range(e)))
            candidates = [x for x in (later_write, later_read) if x is not None]
            later = min(candidates) if candidates else None
            if later is not None and i not in drop:
                what = "rewrite" if later == later_write else "re-read"
                stubs[i] = (
                    f"[{_size(e.get('result')):,} chars read at step {e.get('step')} -- superseded by the "
                    f"{what} at step {history[later].get('step')}; the file is on disk]"
                )
            last_read[(path, _read_range(e))] = i

    return drop, lessons, stubs


def trimmed_view(history: list[dict], window: int) -> list[dict]:
    """The last `window` entries, minus dead material (see module docstring).
    Returns new entry dicts; `history` itself is not modified."""
    drop, lessons, stubs = _decisions(history)
    view = []
    for i in range(max(len(history) - window, 0), len(history)):
        if i in drop:
            continue
        e = dict(history[i])
        if i in stubs:
            if e.get("tool") == "write_file":
                e["args"] = {**e["args"], "content": stubs[i]}
            else:
                e["result"] = stubs[i]
        if i in lessons:
            result = e.get("result")
            e["result"] = {"resolved": lessons[i], **result} if isinstance(result, dict) else f"{lessons[i]} | {result}"
        view.append(e)
    return view


def progress_ledger(history: list[dict]) -> list[str]:
    """One line per script / QE run / file read, over the whole run."""
    scripts: dict[str, dict] = {}
    qe: dict[str, str] = {}
    reads: dict[str, int] = {}

    for e in history:
        tool, path, step = e.get("tool"), _path(e), e.get("step")
        if tool == "write_file" and path:
            s = scripts.setdefault(path, {"fails": 0})
            s.update(written=step, status="written", since=step)
        elif tool == "run_python" and path:
            s = scripts.setdefault(path, {"fails": 0})
            if run_succeeded(e.get("result")):
                s.update(status="ok", since=step, fails_before=s.get("fails", 0), fails=0)
            else:
                if s.get("status") != "failing":
                    s["since"] = step
                s.update(status="failing", fails=s.get("fails", 0) + 1, error=_last_error_line(e.get("result")))
        elif tool == "run_espresso_workflow":
            r = e.get("result")
            directory = (e.get("args") or {}).get("directory", "?")
            ok = isinstance(r, dict) and r.get("success")
            qe[directory] = f"OK at step {step}" if ok else f"FAILED at step {step}: {_last_error_line(r)}"
        elif tool == "read_file" and path:
            reads[path] = step

    lines = []
    for path, s in scripts.items():
        if s["status"] == "ok":
            extra = f" (after {s['fails_before']} failed attempt(s))" if s.get("fails_before") else ""
            lines.append(f"{path}: ran OK at step {s['since']}{extra}")
        elif s["status"] == "failing":
            lines.append(f"{path}: FAILING since step {s['since']} ({s['fails']} attempt(s)); last error: {s['error']}")
        elif path.endswith(".py"):
            lines.append(f"{path}: written at step {s['written']}, not run since")
        else:
            lines.append(f"{path}: written at step {s['written']}")
    lines += [f"QE workflow in {d}: {status}" for d, status in qe.items()]
    if reads:
        lines.append("Files read (last step): " + ", ".join(f"{p} @ {s}" for p, s in reads.items()))
    return lines
