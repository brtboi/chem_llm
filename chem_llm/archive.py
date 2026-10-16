"""A permanent record of every run, kept outside the work directory.

A work dir is cleared at the start of each run (all but *.jsonl), so
anything meant to last lives in the archive instead (settings.archive_dir):

    <archive>/index.jsonl                one line per run, every run ever
    <archive>/runs/<date>_<time>_<case>/
        progress.txt                     the readable live stream (--pretty-progress),
                                         written as the run goes
        summary.txt                      the run summary (chem_llm.report)
        changes.txt                      what changed since the previous run of the
                                         same case: code commits, prompt, settings,
                                         and the result
        record.json                      the run record, format 2 (chem_llm.runlog): the
                                         same fields as the work dir's log.jsonl, with
                                         very long tool results cut
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


def _system_prompt(archive: Path, entry: dict) -> str | None:
    """A run's system prompt, from its record.json (or, for runs archived
    before format 2, prompts/<system_sha>.txt)."""
    try:
        return json.loads((Path(archive) / entry["dir"] / "record.json").read_text())["prompt"]["system"]
    except (OSError, KeyError, TypeError, ValueError):
        pass
    sha = entry.get("system_sha")
    path = Path(archive) / "prompts" / f"{sha}.txt"
    return path.read_text() if sha and path.exists() else None


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
        base = self.archive / "runs" / f"{stamp}_{_slug(case)}"
        (self.archive / "runs").mkdir(parents=True, exist_ok=True)
        # two runs of one case started in the same second must not share a folder
        for n in range(1, 1000):
            self.dir = base if n == 1 else base.with_name(f"{base.name}-{n}")
            try:
                self.dir.mkdir()
                break
            except FileExistsError:
                continue
        self.progress_path = self.dir / "progress.txt"
        self._progress = self.progress_path.open("a", encoding="utf-8")

    @property
    def progress_stream(self):
        return self._progress

    def finish(self, record: dict, summary: str | None, extra: dict | None = None) -> dict:
        """Write record.json / summary / changes and the index line; returns
        the index entry. `record` is the run's log.jsonl record (any format)."""
        from .runlog import for_archive, upgrade

        self._progress.close()
        record = upgrade(record)
        entry = _index_entry(self, record, extra)
        (self.dir / "record.json").write_text(json.dumps(for_archive(record), indent=1, default=str))
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


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in text.lower()).strip("-") or "run"


def _index_entry(run: RunArchive, record: dict, extra: dict | None) -> dict:
    """One index.jsonl line for a format-2 record."""
    from .history import run_succeeded
    from .rundiff import prompt_snapshot
    from .telemetry import operation_times

    times = operation_times(record.get("metrics"))
    cfg = record.get("config") or {}
    prompt = record.get("prompt") or {}
    snap = prompt_snapshot(prompt["system"], prompt["task"] or "") if prompt.get("system") else {}
    steps = record.get("steps") or []
    meta = {k: v for k, v in run.meta.items() if k not in ("compound", "spacegroup", "spacegroup_number")}
    entry = {
        "run_id": record.get("run_id"),
        "dir": str(run.dir.relative_to(run.archive)),
        "time": record.get("timestamp"),
        "case": run.case,
        "label": run.label,
        **meta,
        "backend": cfg.get("backend"),
        "model": record.get("model"),
        "prompt_version": snap.get("version"),
        "system_sha": snap.get("system_sha"),
        "task_sha": snap.get("task_sha"),
        "code": {"commit": cfg.get("code_commit"), "uncommitted_changes": cfg.get("uncommitted_changes") or []},
        "completed": record.get("completed"),
        "band_gap": (record.get("outcome") or {}).get("band_gap"),
        "dft_finished": (record.get("outcome") or {}).get("dft_finished"),
        "steps": record.get("total_steps"),
        "wall_min": round((record.get("runtime") or 0) / 60, 2),
        "llm_min": round(times["llm"] / 60, 2) if times else None,
        "qe_min": round(times["qe"] / 60, 2) if times else None,
        "total_tokens": record.get("total_tokens"),
        "failed_scripts": sum(1 for s in steps if s.get("tool") == "run_python" and not run_succeeded(s.get("result"))),
        "refused": sum(1 for s in steps if s.get("tool") == "refused"),
        "config": {k: cfg.get(k) for k in ("temperature", "do_sample", "max_steps", "max_history", "trim_history")},
    }
    entry.update(extra or {})
    return entry


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
        old_sys = _system_prompt(archive, previous) if archive else None
        new_sys = _system_prompt(archive, current) if archive else None
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
