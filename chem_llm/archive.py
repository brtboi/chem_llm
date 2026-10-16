"""A permanent record of every run, kept outside the work directory.

A work dir is cleared at the start of each run (all but *.jsonl), so
anything meant to last lives in the archive instead (settings.archive_dir):

    <archive>/index.jsonl                one line per run, every run ever
    <archive>/prompts/<sha>.txt          each distinct system prompt and task text,
                                         once (named by record.json's prompt shas)
    <archive>/runs/<date>_<time>_<case>/
        progress.txt                     the readable live stream (--pretty-progress),
                                         written as the run goes
        summary.txt                      the run summary (chem_llm.report)
        changes.txt                      what changed since the previous run of the
                                         same case: code commits, prompt, settings,
                                         and the result
        record.json                      the run: outcome, tokens, metrics, settings,
                                         prompt version, code version, and every step
                                         with its tool call (arguments and result) and
                                         timing -- see archive_record()
        scripts/                         the final version of every script the agent
                                         wrote (top level of the work dir)
        bands.pdf                        the band-structure plot, when there is one

Nothing here is ever deleted by chem_llm. `case` groups runs of the same
job (default: the compound); `label` says why a run was made.
"""
from __future__ import annotations

import contextlib
import difflib
import fcntl
import json
import subprocess
from datetime import datetime
from pathlib import Path

from . import REPO_ROOT


# ----------------------------------------------------------------------
def code_version(repo: Path = REPO_ROOT) -> dict:
    """The git commit the package ran from, and whether the tree had
    uncommitted changes to tracked files ({} outside a git checkout)."""
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=20)

    if not (repo / ".git").exists():
        return {}
    try:
        head = git("rev-parse", "--short=10", "HEAD")
        if head.returncode != 0:
            return {}
        status = git("status", "--porcelain", "--untracked-files=no").stdout.splitlines()
        branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return {}
    changed = [line[3:] for line in status if line.strip()]
    return {"commit": head.stdout.strip(), "branch": branch,
            "uncommitted_changes": [f for f in changed if _is_code(f)]}


def _is_code(path: str) -> bool:
    """Files that change what a run does (not run outputs or notes)."""
    return path.startswith(("chem_llm/", "scripts/")) or path in ("pyproject.toml", "config.example.yaml")


def commits_between(old: str | None, new: str | None, repo: Path = REPO_ROOT) -> list[str] | None:
    """`git log --oneline old..new` (oldest first), or None if unknown."""
    if not (old and new) or old == new:
        return [] if old and old == new else None
    r = subprocess.run(["git", "-C", str(repo), "log", "--reverse", "--format=%h %ad %s", "--date=short",
                        f"{old}..{new}"], capture_output=True, text=True)
    return r.stdout.splitlines() if r.returncode == 0 else None


def commit_at(timestamp: str, repo: Path = REPO_ROOT) -> str | None:
    """The last commit made before `timestamp` (for runs that predate code
    tracking)."""
    r = subprocess.run(["git", "-C", str(repo), "rev-list", "-1", f"--before={timestamp}", "HEAD"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return r.stdout.strip()[:10] or None


# ----------------------------------------------------------------------
@contextlib.contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_index(archive: Path) -> list[dict]:
    index = Path(archive) / "index.jsonl"
    if not index.exists():
        return []
    return [json.loads(line) for line in index.read_text().splitlines() if line.strip()]


def append_index(archive: Path, entry: dict) -> None:
    archive = Path(archive)
    with _locked(archive / ".index.lock"):
        with (archive / "index.jsonl").open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")


def store_prompt(archive: Path, text: str, sha: str) -> Path:
    path = Path(archive) / "prompts" / f"{sha}.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return path


def load_prompt(archive: Path, sha: str | None) -> str | None:
    path = Path(archive) / "prompts" / f"{sha}.txt"
    return path.read_text() if sha and path.exists() else None


# ----------------------------------------------------------------------
class RunArchive:
    """The archive folder of one run. Created when the run starts (so the
    progress stream is on disk from the first step); finish() writes the
    rest once the run's record exists."""

    def __init__(self, archive: Path, case: str, label: str | None = None, meta: dict | None = None):
        self.archive = Path(archive)
        self.case = case
        self.label = label
        self.meta = dict(meta or {})
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.dir = self.archive / "runs" / f"{stamp}_{_slug(case)}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.dir / "progress.txt"
        self._progress = self.progress_path.open("a", encoding="utf-8")

    @property
    def progress_stream(self):
        return self._progress

    def finish(self, record: dict, summary: str | None, extra: dict | None = None) -> dict:
        """Write summary / record / changes and the index line; returns the
        index entry."""
        self._progress.close()
        prompt = record.get("prompt") or {}
        if prompt.get("system") and prompt.get("system_sha"):
            store_prompt(self.archive, prompt["system"], prompt["system_sha"])
        task = (record.get("final_state") or {}).get("task")
        if task and prompt.get("task_sha"):
            store_prompt(self.archive, task, prompt["task_sha"])
        entry = _index_entry(self, record, extra)
        out = archive_record(record, self.case, self.label, self.meta)
        (self.dir / "record.json").write_text(json.dumps(out, indent=1, default=str))
        if summary is not None:
            (self.dir / "summary.txt").write_text(summary)
        self._keep_artifacts(Path(record.get("work_dir") or ""), record.get("outcome") or {})
        previous = next((e for e in reversed(read_index(self.archive)) if e.get("case") == self.case), None)
        (self.dir / "changes.txt").write_text(describe_changes(previous, entry, self.archive))
        append_index(self.archive, entry)
        return entry


    def _keep_artifacts(self, work_dir: Path, outcome: dict) -> None:
        """Copy the agent's scripts and the plot: the work dir is cleared by
        the next run. Small files only (scripts < 200 kB, plot < 5 MB)."""
        import shutil

        if not work_dir.is_dir():
            return
        scripts = [p for p in sorted(work_dir.glob("*.py")) if p.stat().st_size < 200_000]
        if scripts:
            (self.dir / "scripts").mkdir(exist_ok=True)
            for p in scripts:
                shutil.copy2(p, self.dir / "scripts" / p.name)
        plot = Path(outcome["plot"]) if outcome.get("plot") else None
        if plot and plot.is_file() and plot.stat().st_size < 5_000_000:
            shutil.copy2(plot, self.dir / f"bands{plot.suffix}")


RESULT_CAP = 20_000  # characters per string inside a tool result


def _cap(value):
    """Tool results can be long (a whole file read, a QE output tail);
    strings beyond RESULT_CAP are cut, with the cut stated."""
    if isinstance(value, str) and len(value) > RESULT_CAP:
        return value[:RESULT_CAP] + f" ... [{len(value) - RESULT_CAP:,} more characters cut in the archive]"
    if isinstance(value, dict):
        return {k: _cap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cap(v) for v in value]
    return value


def archive_record(record: dict, case: str, label: str | None, meta: dict | None = None) -> dict:
    """The archived record.json of one run, built from its log.jsonl record.

    Top level: what ran (run_id, case, label, suite, purpose, model, backend),
    how it went (completed, num_steps, wall_s, tokens, outcome, final_result),
    and why (prompt, config, code). `steps` is every agent step in order: the
    tool it called with the full arguments, the result, and the step's model
    and tool time and tokens. The prompt texts themselves are in
    prompts/<system_sha>.txt and prompts/<task_sha>.txt next to runs/.
    """
    fs = record.get("final_state") or {}
    prompt = record.get("prompt") or {}
    metrics = dict(record.get("metrics") or {})
    llm = metrics.get("llm") or {}
    p_tok = (llm.get("prompt_tokens") or {}).get("total")
    c_tok = (llm.get("completion_tokens") or {}).get("total")
    code = dict(record.get("code") or {})
    if "dirty" in code:  # runs logged before 2026-10-16
        code["uncommitted_changes"] = [f for f in code.pop("dirty") or [] if _is_code(f)]
    meta = dict(meta or {})

    history = fs.get("history") or []
    by_index = {e.get("step"): e for e in history}
    steps = []
    timing = fs.get("steps") or []
    for rec in timing:
        entries = [by_index[i] for i in rec.get("history", []) if i in by_index]
        main = entries[-1] if entries else {}
        step = {"step": rec.get("step"), "tool": main.get("tool", rec.get("tool")),
                "args": main.get("args"), "result": _cap(main.get("result"))}
        if len(entries) > 1:
            step["also_logged"] = [{"tool": e["tool"], "args": e.get("args"), "result": _cap(e.get("result"))}
                                   for e in entries[:-1]]
        for key in ("llm_s", "tool_s", "prompt_tokens", "completion_tokens", "t_start", "t_end"):
            if rec.get(key) is not None:
                step[key] = rec[key]
        steps.append(step)
    if not timing:  # runs logged before per-step timing existed
        steps = [{"step": e.get("step"), "tool": e.get("tool"), "args": e.get("args"), "result": _cap(e.get("result"))}
                 for e in history]

    return {
        "run_id": record.get("run_id"),
        "timestamp": record.get("timestamp"),
        "case": case,
        "label": label,
        "suite": meta.pop("suite", None),
        "purpose": meta.pop("purpose", None),
        **meta,
        "model": record.get("model"),
        "backend": (record.get("config") or {}).get("backend"),
        "completed": record.get("completed"),
        "num_steps": len(history) or record.get("num_steps"),
        "wall_s": metrics.get("wall_s", round(record.get("runtime") or 0, 1)),
        "total_tokens": (p_tok or 0) + (c_tok or 0) if (p_tok or c_tok) else None,
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "outcome": record.get("outcome"),
        "final_result": fs.get("final_result"),
        "prompt": {
            "version": prompt.get("version"),
            "system_sha": prompt.get("system_sha"),
            "task_sha": prompt.get("task_sha"),
            "system_file": f"prompts/{prompt['system_sha']}.txt" if prompt.get("system_sha") else None,
            "task_file": f"prompts/{prompt['task_sha']}.txt" if prompt.get("task_sha") else None,
        },
        "metrics": metrics,
        "config": record.get("config"),
        "code": code,
        "work_dir": record.get("work_dir"),
        "steps": steps,
    }


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in text.lower()).strip("-") or "run"


def _index_entry(run: RunArchive, record: dict, extra: dict | None) -> dict:
    m = record.get("metrics") or {}
    out = record.get("outcome") or {}
    cfg = record.get("config") or {}
    hist = (record.get("final_state") or {}).get("history", [])
    phases = m.get("phases_s") or {}
    entry = {
        "run_id": record.get("run_id"),
        "dir": str(run.dir.relative_to(run.archive)),
        "time": record.get("timestamp"),
        "case": run.case,
        "label": run.label,
        **run.meta,
        "backend": cfg.get("backend"),
        "model": record.get("model"),
        "prompt_version": (record.get("prompt") or {}).get("version"),
        "system_sha": (record.get("prompt") or {}).get("system_sha"),
        "task_sha": (record.get("prompt") or {}).get("task_sha"),
        "code": record.get("code") or {},
        "completed": record.get("completed"),
        "band_gap": out.get("band_gap"),
        "dft_finished": out.get("dft_finished"),
        "steps": len(hist),
        "wall_min": round((record.get("runtime") or 0) / 60, 2),
        "llm_min": round(phases.get("llm", 0) / 60, 2) if phases else None,
        "qe_min": round(phases.get("qe", phases.get("quantum_espresso", 0)) / 60, 2) if phases else None,
        "failed_scripts": sum(1 for e in hist if e["tool"] == "run_python"
                              and isinstance(e.get("result"), dict) and e["result"].get("returncode") not in (0, None)),
        "refused": sum(1 for e in hist if e["tool"] == "refused"),
        "config": {k: cfg.get(k) for k in ("decoding", "max_steps", "max_history", "trim_history")},
    }
    entry.update(extra or {})
    return entry


# ----------------------------------------------------------------------
def _fmt(v, unit="") -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}{unit}" if unit == " eV" else f"{v:g}{unit}"
    return f"{v}{unit}"


def describe_changes(previous: dict | None, current: dict, archive: Path | None = None) -> str:
    """Readable 'what changed since last time' for one case."""
    case = current.get("case")
    if previous is None:
        return f"First recorded run of {case}: nothing to compare against.\n"
    lines = [f"Changes since the previous {case} run ({(previous.get('time') or '?')[:16]}, "
             f"{previous.get('dir') or previous.get('run_id')})", ""]

    old_c, new_c = (previous.get("code") or {}), (current.get("code") or {})
    commits = commits_between(old_c.get("commit"), new_c.get("commit"))
    lines.append(f"code: {old_c.get('commit', '?')}{' (inferred from the run time)' if old_c.get('inferred') else ''}"
                 f" -> {new_c.get('commit', '?')}")
    if commits:
        lines.append(f"  {len(commits)} commit(s) in between:")
        lines += [f"    {c}" for c in commits[-40:]]
    elif commits == []:
        lines.append("  same commit")
    uncommitted = new_c.get("uncommitted_changes", [f for f in new_c.get("dirty") or [] if _is_code(f)])
    if uncommitted:
        lines.append(f"  uncommitted changes to the code at run time: {', '.join(uncommitted[:12])}")

    lines.append("")
    if previous.get("prompt_version") and previous.get("prompt_version") == current.get("prompt_version"):
        lines.append(f"prompt: unchanged (version {current.get('prompt_version')})")
    else:
        lines.append(f"prompt: version {previous.get('prompt_version') or 'not recorded'} -> {current.get('prompt_version')}")
        if previous.get("task_sha") and previous.get("task_sha") != current.get("task_sha"):
            lines.append("  task text changed")
        old_sys = load_prompt(archive, previous.get("system_sha")) if archive else None
        new_sys = load_prompt(archive, current.get("system_sha")) if archive else None
        if old_sys and new_sys and old_sys != new_sys:
            diff = [d for d in difflib.unified_diff(old_sys.splitlines(), new_sys.splitlines(), n=0, lineterm="")
                    if d[:1] in "+-" and not d.startswith(("+++", "---"))]
            lines.append(f"  system prompt: {sum(d[0] == '+' for d in diff)} line(s) added, "
                         f"{sum(d[0] == '-' for d in diff)} removed:")
            lines += [f"    {d[:200]}" for d in diff[:60]]
            if len(diff) > 60:
                lines.append(f"    ... {len(diff) - 60} more")

    old_cfg, new_cfg = previous.get("config") or {}, current.get("config") or {}
    changed = [k for k in sorted(set(old_cfg) | set(new_cfg)) if old_cfg.get(k) != new_cfg.get(k)]
    if previous.get("backend") != current.get("backend"):
        changed.insert(0, "backend")
    lines.append("")
    if changed:
        lines.append("settings:")
        for k in changed:
            a = previous.get(k) if k == "backend" else old_cfg.get(k)
            b = current.get(k) if k == "backend" else new_cfg.get(k)
            lines.append(f"  {k}: {_fmt(a)} -> {_fmt(b)}")
    else:
        lines.append("settings: unchanged")

    lines += ["", "result:"]
    for key, label, unit in (("completed", "completed", ""), ("passed", "benchmark checks passed", ""),
                             ("band_gap", "band gap", " eV"), ("steps", "steps", ""),
                             ("wall_min", "wall time", " min"), ("llm_min", "model time", " min"),
                             ("failed_scripts", "failed script runs", ""), ("refused", "refused calls", "")):
        if key in previous or key in current:
            a, b = previous.get(key), current.get(key)
            mark = "" if _fmt(a) == _fmt(b) else "   *"
            lines.append(f"  {label:<24} {_fmt(a, unit):>12} -> {_fmt(b, unit):<12}{mark}")
    return "\n".join(lines) + "\n"
