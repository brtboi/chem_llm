"""A readable live view of an agent run: one plain-language line per action.

The raw stream prints every step's model output, tool call and full tool
result -- JSON, file contents, tracebacks. ProgressPrinter instead says
what the agent is doing as it starts each action, and how it went when it
finishes:

    [ 2 | 00:41] fetching the TiO2 (P4_2/mnm) structure from the Materials Project
                 -> mp-2657, saved structures/TiO2_base.cif
    [ 3 | 01:12] note: Primitive cell has 6 atoms; next, reduce with seekpath ...
    [16 | 09:30] fetching a norm-conserving, scalar-relativistic PBEsol pseudopotential
                 (stringent) for Ti -> z_valence 12, recommended ecutwfc 42 Ha
    [21 | 13:02] running setup_jobs.py -> FAILED: NameError: name 'Ecutwfc_RY' is not defined

Everything is formatted from the tool call and its result; the model does
nothing differently. Use it through `chem-llm --pretty-progress`, or pass an
instance as `DFTAgent.run(on_step=...)`.
"""
from __future__ import annotations

import sys
import textwrap
import time
from pathlib import Path

from .history import run_succeeded

__all__ = ["ProgressPrinter", "describe_call", "describe_result"]

_RELATIVITY = {"sr": "scalar-relativistic", "fr": "fully relativistic"}
_KIND = {"nc": "norm-conserving", "paw": "PAW"}
_FUNCTIONAL = {"pbe": "PBE", "pbesol": "PBEsol", "pw": "LDA (PW)", "lda": "LDA"}


def _name(path) -> str:
    return Path(str(path)).name if path else "?"


def _last_line(text, limit: int = 120) -> str:
    lines = [l.strip() for l in str(text or "").strip().splitlines() if l.strip()]
    return (lines[-1] if lines else "")[:limit]


def _one_line(text, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def describe_call(call: dict) -> str:
    """What the agent is about to do, in plain words."""
    tool, a = call.get("tool"), call.get("args") or {}
    if tool == "generate_cif":
        sg = a.get("spacegroup_symbol") or (f"#{a['spacegroup_number']}" if a.get("spacegroup_number") else None)
        return f"fetching the {a.get('composition', '?')}{f' ({sg})' if sg else ''} structure from the Materials Project"
    if tool == "get_pseudopotential":
        desc = ", ".join(x for x in (_KIND.get(a.get("kind"), a.get("kind")),
                                    _RELATIVITY.get(a.get("relativity"), a.get("relativity"))) if x)
        func = _FUNCTIONAL.get(str(a.get("generator", "")).lower(), a.get("generator"))
        acc = f" ({a['accuracy']})" if a.get("accuracy") else ""
        return f"fetching a {desc} {func} pseudopotential{acc} for {a.get('element', '?')}"
    if tool == "write_file":
        return f"writing {_name(a.get('path'))} ({len(a.get('content') or ''):,} chars)"
    if tool == "run_python":
        return f"running {_name(a.get('path'))}"
    if tool == "read_file":
        if a.get("start_line") or a.get("num_lines"):
            start = int(a.get("start_line") or 1)
            rng = f"lines {start}-{start + int(a['num_lines']) - 1}" if a.get("num_lines") else f"from line {start}"
            return f"reading {_name(a.get('path'))} ({rng})"
        return f"reading {_name(a.get('path'))}"
    if tool == "make_dir":
        return f"creating directory {a.get('path', '?')}"
    if tool == "search_docs":
        return f"searching the documentation: \"{_one_line(a.get('query'), 70)}\""
    if tool == "validate_qe_input":
        return f"validating the QE input {a.get('path', '?')}"
    if tool == "get_kpath":
        return f"getting the standard k-path for {_name(a.get('structure_path'))}"
    if tool == "run_espresso_workflow":
        return f"running Quantum ESPRESSO in {a.get('directory', '?')} (SCF, bands, bands.x)"
    if tool == "train_deepseudopot":
        return "training the DeePseudopot model"
    if tool in ("note", "done"):
        return ""  # shown with their text when they finish
    return f"{tool}({_one_line(a, 60)})"


def describe_result(entry: dict) -> str:
    """How a logged step turned out, in a few words ('' if nothing to add)."""
    tool, a, r = entry.get("tool"), entry.get("args") or {}, entry.get("result")
    ok = isinstance(r, dict) and r.get("success") is not False
    failed = (isinstance(r, str) and r.startswith("ERROR")) or (isinstance(r, dict) and r.get("success") is False)
    if failed:
        if isinstance(r, dict) and r.get("issues"):
            more = f" (+{len(r['issues']) - 1} more)" if len(r["issues"]) > 1 else ""
            return f"FAILED: {_one_line(r['issues'][0], 160)}{more}"
        err = r if isinstance(r, str) else r.get("stderr") or r
        return f"FAILED: {_last_line(err, 160)}"
    if tool == "generate_cif" and ok:
        gap = f", MP gap {r['band_gap_eV']:.2f} eV" if isinstance(r.get("band_gap_eV"), (int, float)) else ""
        return f"{r.get('selected_material_id')}{gap}, saved {_name(r.get('output_path'))}"
    if tool == "get_kpath" and ok:
        return f"{r.get('spacegroup')}: {r.get('path')} ({r.get('n_kpoints')} k-points)"
    if tool == "get_pseudopotential" and ok:
        f, rec = r.get("file") or {}, r.get("recommended_ecutwfc") or {}
        parts = [f"z_valence {f['z_valence']:g}" if f.get("z_valence") is not None else "",
                 f"recommended ecutwfc {rec['value']:g} {rec.get('unit', 'Ha')}" if rec.get("value") else "",
                 f"WARNING: {r['file_warning']}" if r.get("file_warning") else ""]
        return ", ".join(p for p in parts if p)
    if tool == "run_python":
        if run_succeeded(r):
            out = _last_line(r.get("stdout")) if isinstance(r, dict) else ""
            return "OK" + (f" ({_one_line(out, 80)})" if out else "")
        return f"FAILED: {_last_line(r.get('stderr') if isinstance(r, dict) else r)}"
    if tool == "read_file" and isinstance(r, dict):
        extra = f"; {r['stale_warning'].split(':')[0]}" if r.get("stale_warning") else ""
        return f"{r.get('end_line', 0) - r.get('start_line', 1) + 1} of {r.get('total_lines', '?')} lines{extra}"
    if tool == "search_docs" and isinstance(r, dict):
        titles = [x.get("title", "") for x in r.get("results", [])[:2]]
        return f"{r.get('num_results', len(r.get('results', [])))} results" + (f": {'; '.join(_one_line(t, 40) for t in titles)}" if titles else "")
    if tool == "validate_qe_input" and isinstance(r, dict):
        return "OK" if not r.get("issues") else f"issues: {_one_line(r['issues'][0], 110)}"
    if tool == "run_espresso_workflow" and isinstance(r, dict):
        return f"OK ({', '.join(r.get('completed_steps', []))})"
    if tool == "make_dir":
        return ""
    return ""


class Observers:
    """Several loop observers as one (e.g. the caller's printer plus the
    archive's progress.txt)."""

    def __init__(self, *observers):
        self.observers = [o for o in observers if o is not None]

    def start(self, step: int, call: dict) -> None:
        for o in self.observers:
            o.start(step, call)

    def finish(self, step: int, entry: dict) -> None:
        for o in self.observers:
            o.finish(step, entry)


class ProgressPrinter:
    """Observer for the agent loop (run_step_loop) that prints readable lines.

    `start` prints what an action is before it runs, so a long QE run shows
    what is happening while it happens; `finish` adds the outcome. Notes are
    printed with their text, wrapped; very long notes are shortened.
    """

    def __init__(self, stream=None, width: int = 100, note_chars: int = 180):
        self.stream = stream or sys.stdout
        self.width = width
        self.note_chars = note_chars
        self.t0 = time.monotonic()

    def _prefix(self, step: int) -> str:
        m, s = divmod(int(time.monotonic() - self.t0), 60)
        h, m = divmod(m, 60)
        clock = f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        return f"[{step:>2} | {clock}] "

    def _write(self, prefix: str, text: str) -> None:
        indent = " " * len(prefix)
        lines = textwrap.wrap(text, self.width, initial_indent=prefix, subsequent_indent=indent) or [prefix]
        print("\n".join(lines), file=self.stream, flush=True)

    def start(self, step: int, call: dict) -> None:
        text = describe_call(call)
        if text:
            self._write(self._prefix(step), text + " ...")

    def finish(self, step: int, entry: dict) -> None:
        tool, args = entry.get("tool"), entry.get("args") or {}
        prefix = self._prefix(step)
        if tool == "note":
            self._write(prefix, f"note: {_one_line(args.get('text'), self.note_chars)}")
        elif tool == "done":
            self._write(prefix, f"done: {_one_line(args.get('summary'), 200)}")
        elif tool == "refused":
            self._write(prefix, f"harness refused a {args.get('tool')} call: {_one_line(entry.get('result'), 120)}")
        elif tool == "parse_error":
            self._write(prefix, "the model's output was not a valid tool call; asking again")
        elif tool == "api_error":
            self._write(prefix, f"API error: {_last_line(entry.get('result'))}")
        else:
            outcome = describe_result(entry)
            if outcome:
                self._write(" " * len(prefix), f"-> {outcome}")
