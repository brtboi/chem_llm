"""A readable summary of a finished agent run, built deterministically.

Everything comes from the run's own record -- the tool calls and their
results, the scripts it wrote (their logged contents), its notes and done
summary -- plus the QE inputs and outputs left in the work directory. No
model is involved, so nothing in the summary can be invented: a value is
either read from a file or a tool result, or it is reported as missing.

    from chem_llm.report import summarize_log
    print(summarize_log("sandbox/test29"))          # latest run logged there

Sections:
  1. What the agent did -- data pulled from outside sources, scripts written
     and run, Quantum ESPRESSO runs, in order.
  2. Results and key choices, in the order they were made -- structure,
     displacements, pseudopotentials, cutoffs/k-mesh, k-path, DFT, band gap.
  3. Uncertainties and difficulties -- failed attempts and their errors,
     where the documentation lookups clustered, harness refusals, notes that
     flag doubt, and the agent's own stated assumptions / review items.
"""
from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

from .history import run_succeeded

__all__ = ["summarize_run", "summarize_log", "load_log_entry"]

WIDTH = 100
_DOUBT = re.compile(
    r"\b(not sure|unsure|uncertain|unclear|not certain|unknown whether|assum\w*|approximat\w*|"
    r"should be (?:checked|verified|reviewed)|double[- ]check|verify|expert|caveat|"
    r"may need|might (?:be|need)|could be wrong|questionable|risk\w*|workaround|unexpected)\b",
    re.I,
)


@dataclass
class _Item:
    step: int | None
    text: str


# ----------------------------------------------------------------------------
# small helpers

def _name(path) -> str:
    return Path(str(path)).name


def _rel(path, work_dir: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(work_dir.resolve()))
    except (ValueError, OSError):
        return str(path)


def _last_line(text: str, limit: int = 140) -> str:
    lines = [l.strip() for l in str(text).strip().splitlines() if l.strip()]
    return (lines[-1] if lines else "")[:limit]


def _wrap(text: str, indent: int, first: str = "") -> list[str]:
    pad = " " * indent
    return textwrap.wrap(text, WIDTH, initial_indent=pad + first, subsequent_indent=pad + " " * len(first)) or [pad + first]


def _step_label(step) -> str:
    return f"step {step:>2}" if isinstance(step, int) else "       "


def _read(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


def _upf_facts(path) -> dict:
    """Header facts of a pseudopotential file still on disk, else {}."""
    from .tools.upf import read_upf_header
    try:
        return read_upf_header(path) if path else {}
    except (OSError, ValueError):
        return {}


def _namelist_values(text: str) -> dict:
    vals = {}
    for key in ("ecutwfc", "ecutrho", "nbnd", "occupations", "conv_thr", "noncolin", "lspinorb", "smearing", "degauss", "nspin"):
        m = re.search(rf"^\s*{key}\s*=\s*([^,\n]+)", text, re.M | re.I)
        if m:
            vals[key] = m.group(1).strip().strip("'\"")
    return vals


# ----------------------------------------------------------------------------
# extraction

def _scripts(history: list[dict]) -> dict[str, dict]:
    """Per python script: first/last write, versions, runs, failures, status."""
    scripts: dict[str, dict] = {}
    for e in history:
        path = (e.get("args") or {}).get("path", "")
        if not str(path).endswith(".py"):
            continue
        s = scripts.setdefault(str(path), {"first": e["step"], "writes": 0, "runs": 0, "fails": [], "ok_step": None,
                                           "content": None, "status": "written"})
        if e["tool"] == "write_file":
            s["writes"] += 1
            s["writes_since_run"] = s.get("writes_since_run", 0) + 1
            s["content"] = e["args"].get("content")
            s["status"] = "written, not run since"
        elif e["tool"] == "run_python":
            s["runs"] += 1
            s["writes_since_run"] = 0
            if run_succeeded(e.get("result")):
                s["ok_step"], s["status"] = e["step"], "ok"
            else:
                r = e.get("result")
                err = r.get("stderr", "") if isinstance(r, dict) else str(r)
                s["fails"].append((e["step"], _last_line(err)))
                s["status"] = "failing"
    return scripts


def _structure_script(scripts: dict[str, dict]) -> tuple[str, dict] | None:
    """The script that wrote the perturbed structures: mentions displacements."""
    for path, s in scripts.items():
        if s["content"] and re.search(r"displac|rattle|perturb", s["content"], re.I) and re.search(r"structure_|\.cif", s["content"]):
            return path, s
    return None


def _displacement_lines(content: str) -> list[str]:
    """Numeric parameter assignments about displacements/strain/temperature."""
    out = []
    for line in content.splitlines():
        m = re.match(r"^\s*([A-Za-z_]\w*)\s*=\s*([-+]?\d[\d.eE+-]*)\s*(#.*)?$", line)
        if m and re.search(r"disp|sigma|ampl|strain|temp|rattle|std|stdev|scale", m.group(1), re.I):
            out.append(f"{m.group(1)} = {m.group(2)}" + (f"  {m.group(3).strip()}" if m.group(3) else ""))
    return out[:6]


def _kpath(bands_in: str | None, labels_txt: str | None) -> dict:
    info: dict = {}
    if bands_in:
        m = re.search(r"K_POINTS\s*\{?\(?\s*(\w+)", bands_in)
        info["kind"] = m.group(1) if m else None
        block = bands_in[m.end():] if m else ""
        lines = block.strip().splitlines()
        if lines and lines[0].strip().isdigit():
            n = int(lines[0])
            info["n"] = n
            pts = []
            for l in lines[1:n + 1]:
                parts = l.split()
                try:
                    pts.append(tuple(float(x) for x in parts[:3]))
                except ValueError:
                    break
            info["points"] = pts
    if labels_txt:
        found = []
        for l in labels_txt.splitlines():
            if l.strip().startswith("#"):
                continue
            toks = l.replace(",", " ").split()
            ints = [t for t in toks if re.fullmatch(r"\d+", t)]
            names = [t for t in toks if not re.fullmatch(r"[-\d.]+", t)]
            if len(ints) == 1 and names:
                found.append((int(ints[0]), names[0]))
        info["labels"] = sorted(found)
    pts = info.get("points") or []
    if info.get("kind", "").lower() in ("crystal", "tpiba") and len(pts) > 2:
        import math
        d = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        positive = sorted(x for x in d if x > 1e-9)
        typical = positive[len(positive) // 2] if positive else 0
        info["breaks"] = [i + 1 for i, x in enumerate(d) if typical and x > 3 * typical]
        # Segment endpoints: first/last point, every turn, and both sides of a
        # break -- each one is a high-symmetry point that should be labelled.
        corners = {0, len(pts) - 1}
        for b in info["breaks"]:
            corners |= {b - 1, b}
        unit = lambda a, b: tuple((y - x) / (math.dist(a, b) or 1) for x, y in zip(a, b))
        for i in range(1, len(pts) - 1):
            if i in info["breaks"] or i + 1 in info["breaks"]:
                continue
            u, w = unit(pts[i - 1], pts[i]), unit(pts[i], pts[i + 1])
            if math.dist(u, w) > 1e-3 and d[i - 1] > 1e-9 and d[i] > 1e-9:
                corners.add(i)
        info["corners"] = sorted(corners)
    return info


def _doc_clusters(history: list[dict]) -> list[tuple[str, list[dict]]]:
    """Group search_docs calls by what the agent did next (the script it then
    wrote/ran, or the tool it then called)."""
    groups: list[tuple[str, list[dict]]] = []
    pending: list[dict] = []
    for e in history:
        if e["tool"] == "search_docs":
            pending.append(e)
            continue
        if pending and e["tool"] not in ("note", "refused", "parse_error", "api_error", "read_file"):
            target = _name(e["args"].get("path")) if e["args"].get("path") else e["tool"]
            groups.append((target, pending))
            pending = []
    if pending:
        groups.append(("(end of run)", pending))
    return groups


def _summary_sections(text: str | None) -> dict[str, str]:
    """The done summary's 'HEADING:' sections (newer runs write them)."""
    if not text:
        return {}
    parts = re.split(r"(?m)^\s*([A-Z][A-Z0-9 /&()'-]{3,}):", text)
    return {parts[i].strip(): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def _doubt_sentences(texts: list[tuple[int, str]], limit: int = 6) -> list[tuple[int, str]]:
    seen, out = set(), []
    for step, text in texts:
        for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
            if _DOUBT.search(sent) and len(sent) > 25:
                key = re.sub(r"\W+", " ", sent.lower())[:80]
                if key not in seen:
                    seen.add(key)
                    out.append((step, sent.strip()[:230]))
    return out[:limit]


# ----------------------------------------------------------------------------
# the report

def summarize_run(
    history: list[dict],
    work_dir: str | Path,
    *,
    final_result: str | None = None,
    completed: bool = False,
    runtime_seconds: float | None = None,
    model: str | None = None,
    finished_at: float | None = None,
    compound: str | None = None,
    spacegroup: str | None = None,
) -> str:
    """Plain-text summary of one run. `history` is AgentState.history.

    finished_at: the run's end time (epoch seconds). When given, files in
    the work dir modified after it belong to a later run and are ignored.
    """
    from .pipeline import _inspect_outputs

    work_dir = Path(work_dir)
    scripts = _scripts(history)
    by_tool = lambda t: [e for e in history if e["tool"] == t]
    cifs, pseudos, qe_runs = by_tool("generate_cif"), by_tool("get_pseudopotential"), by_tool("run_espresso_workflow")
    searches, refusals, parse_errors = by_tool("search_docs"), by_tool("refused"), by_tool("parse_error")

    compound = next((e["args"].get("composition") for e in cifs), None) or compound
    spacegroup = next((e["args"].get("spacegroup_symbol") for e in cifs), None) or spacegroup

    # the calculation directory the QE workflow ran in (default calculations/000)
    calc_rel = next((e["args"].get("directory") for e in reversed(qe_runs) if e["args"].get("directory")), "calculations/000")
    calc = Path(calc_rel) if Path(calc_rel).is_absolute() else work_dir / calc_rel
    stale = finished_at is not None and any(
        f.stat().st_mtime > finished_at + 60 for f in calc.glob("*") if f.is_file())
    found = _inspect_outputs(work_dir, job=calc.name) if calc.parent.name == "calculations" and not stale else {}
    gap = found.get("band_gap")

    lines: list[str] = []
    title = f"chem_llm run summary -- {compound or 'unknown compound'}" + (f" ({spacegroup})" if spacegroup else "")
    lines += [title, "=" * len(title)]
    status = "completed" if completed else "DID NOT COMPLETE"
    meta = f"{status} in {len(history)} steps" + (f", {runtime_seconds / 60:.1f} min" if runtime_seconds else "") + (f" | {model}" if model else "")
    lines.append(meta)
    lines.append(f"work dir: {work_dir}")
    if gap is not None:
        lines.append(f"band gap: {gap:.3f} eV (computed from the bands output)")
    if found.get("plot"):
        lines.append(f"plot:     {_rel(found['plot'], work_dir)}")
    if stale:
        lines += _wrap(f"the files in {_rel(calc, work_dir)} were written by a later run in this work dir, so "
                       "values read from QE inputs/outputs are omitted below", 0, "note:     ")

    # -- 1. what the agent did ------------------------------------------------
    did: list[_Item] = []
    for e in cifs:
        r = e.get("result") or {}
        sg = r.get("space_group", {}) if isinstance(r, dict) else {}
        if isinstance(r, dict) and r.get("success"):
            did.append(_Item(e["step"], f"Materials Project: {e['args'].get('composition')} "
                                        f"({sg.get('symbol', e['args'].get('spacegroup_symbol'))}) -> {r.get('selected_material_id')}, "
                                        f"saved {_rel(r.get('output_path', ''), work_dir)}"))
        else:
            did.append(_Item(e["step"], f"Materials Project lookup FAILED: {_last_line(r.get('stderr', r) if isinstance(r, dict) else r)}"))
    for e in pseudos:
        a, r = e["args"], e.get("result") or {}
        if not (isinstance(r, dict) and r.get("success")):
            did.append(_Item(e["step"], f"Pseudo-Dojo {a.get('element')}: FAILED -- {_last_line(r.get('stderr', '') if isinstance(r, dict) else r)}"))
            continue
        f = r.get("file") or _upf_facts(r.get("output_path"))  # older runs: read the file on disk
        rec = r.get("recommended_ecutwfc") or {}
        hints = r.get("hints") or {}  # older runs
        z = (f"z_valence {f['z_valence']:g}" if f.get("z_valence") is not None
             else "z_valence unknown (not in the tool result; file no longer on disk)")
        cut = (f"recommended ecutwfc {rec.get('value')} {rec.get('unit', 'Ha')}" if rec
               else f"cutoff hint {hints.get('ecut')} {hints.get('ecut_unit', 'Ha')} (old tool: may not match the file)" if hints else "no cutoff hint")
        did.append(_Item(e["step"], f"Pseudo-Dojo: {a.get('element')} pseudopotential ({a.get('kind', '?')}, "
                                    f"{a.get('relativity', '?')}, {a.get('generator', '?')}, {a.get('accuracy', '?')}) -> {z}, {cut}"))
    for path, s in scripts.items():
        runs = f"run {s['runs']}x" + (f" ({len(s['fails'])} failed)" if s["fails"] else "")
        end = (f"OK at step {s['ok_step']}" if s["status"] == "ok"
               else f"FAILING since step {s['fails'][-1][0]}" if s["status"] == "failing" else s["status"])
        versions = f"{s['writes']} version{'s' if s['writes'] != 1 else ''}"
        did.append(_Item(s["first"], f"wrote {_name(path)}: {versions}, {runs} -> {end}"))
    for e in qe_runs:
        r = e.get("result") or {}
        ok = isinstance(r, dict) and r.get("success")
        steps = ", ".join(r.get("completed_steps", [])) if isinstance(r, dict) else ""
        did.append(_Item(e["step"], f"Quantum ESPRESSO on {_rel(e['args'].get('directory', ''), work_dir)}: "
                                    + (f"OK ({steps})" if ok else f"FAILED -- {_last_line(r.get('stderr', '') if isinstance(r, dict) else r)}")))
    for e in by_tool("validate_qe_input"):
        r = e.get("result") or {}
        issues = r.get("issues") if isinstance(r, dict) else None
        did.append(_Item(e["step"], f"validated {_rel(e['args'].get('path', ''), work_dir)}: "
                                    + ("OK" if isinstance(r, dict) and r.get("success") else f"issues {issues}")))
    if searches:
        did.append(_Item(None, f"documentation lookups: {len(searches)} (where they clustered: section 3)"))

    lines += ["", "1. What the agent did"]
    for it in sorted(did, key=lambda i: (i.step is None, i.step or 0)):
        lines += _wrap(it.text, 4, f"{_step_label(it.step)}  ")

    # -- 2. results and key choices, chronologically ----------------------------
    chosen: list[_Item] = []
    for e in cifs:
        r = e.get("result") or {}
        if isinstance(r, dict) and r.get("output_path"):
            try:
                from ase.io import read
                n_base = len(read(r["output_path"]))
            except Exception:
                n_base = None
            n_dft = found.get("n_atoms")
            if n_base or n_dft:
                chosen.append(_Item(e["step"], "structure: " + (f"{n_base} atoms in the Materials Project cell" if n_base else "")
                                    + (f" -> {n_dft} atoms in the cell used for DFT" if n_dft else "")))
    gen = _structure_script(scripts)
    if gen:
        path, s = gen
        disp = _displacement_lines(s["content"] or "")
        chosen.append(_Item(s["ok_step"] or s["first"], f"perturbed structures ({_name(path)}): "
                            + ("; ".join(disp) if disp else "no numeric displacement parameter found in the script")))
    rels = sorted({e["args"].get("relativity") for e in pseudos if isinstance(e.get("result"), dict) and e["result"].get("success")} - {None})
    if pseudos:
        word = {"sr": "scalar-relativistic", "fr": "fully relativistic (spin-orbit)"}
        zv = [(e["args"].get("element"), (e["result"].get("file") or _upf_facts(e["result"].get("output_path"))).get("z_valence"))
              for e in pseudos if isinstance(e.get("result"), dict) and e["result"].get("success")]
        zs = ", ".join(f"{el} {z:g}" for el, z in zv if z is not None)
        chosen.append(_Item(pseudos[0]["step"], "pseudopotentials: " + (" + ".join(word.get(r, r) for r in rels) or "?")
                            + (f"; valence electrons per atom: {zs}" if zs else "")))
    kpath_info: dict = {}
    pw_in = None if stale else _read(calc / "pw.in")
    if pw_in:
        v = _namelist_values(pw_in)
        recs = [((e.get("result") or {}).get("recommended_ecutwfc") or {}).get("value") for e in pseudos]
        recs = [x for x in recs if isinstance(x, (int, float))]
        rec_txt = f" (largest recommended: {max(recs):g} Ha = {2 * max(recs):g} Ry)" if recs else ""
        mesh = re.search(r"K_POINTS\s*\{?\s*automatic\s*\}?\s*\n\s*([\d ]+)", pw_in, re.I)
        before_qe = [e["step"] for e in history if e["tool"] == "run_python" and run_succeeded(e.get("result"))
                     and (not qe_runs or e["step"] < qe_runs[-1]["step"])]
        step = before_qe[-1] if before_qe else None
        chosen.append(_Item(step, f"cutoffs: ecutwfc {v.get('ecutwfc', '?')} Ry, ecutrho {v.get('ecutrho', '?')} Ry{rec_txt}"))
        scf = [f"occupations {v['occupations']}" if "occupations" in v else None,
               f"k-mesh {' x '.join(mesh.group(1).split()[:3])} (shift {' '.join(mesh.group(1).split()[3:]) or '0 0 0'})" if mesh else None,
               f"conv_thr {v['conv_thr']}" if "conv_thr" in v else None,
               "spin-orbit ON (noncolin, lspinorb)" if v.get("lspinorb", "").lower().strip(".") in ("true", "t") else None]
        chosen.append(_Item(step, "SCF settings: " + ", ".join(x for x in scf if x)))
        kp = _kpath(_read(calc / "bands.in"), _read(calc / "band_labels.dat"))
        kpath_info = kp
        bands_nbnd = _namelist_values(_read(calc / "bands.in") or "").get("nbnd")
        if kp:
            labels = " ".join(name for _, name in kp.get("labels", [])) or "no labels file"
            breaks = kp.get("breaks")
            txt = (f"k-path: {labels}; {kp.get('n', '?')} points ({kp.get('kind', '?')})"
                   + (f", {len(breaks)} discontinuit{'y' if len(breaks) == 1 else 'ies'}" if breaks else "")
                   + (f"; nbnd {bands_nbnd} in the bands run" if bands_nbnd else ""))
            chosen.append(_Item(step, txt))
    if qe_runs and found.get("scf_converged") is not None and pw_in:
        dft = []
        if found.get("n_electrons"):
            dft.append(f"{found['n_electrons']:g} electrons")
        dft.append("SCF converged" if found.get("scf_converged") else "SCF NOT converged")
        dft.append("all three QE stages finished" if found.get("dft_finished") else "QE stages incomplete")
        chosen.append(_Item(qe_runs[-1]["step"], "DFT: " + ", ".join(dft)))
    reported = None
    for text in [final_result or ""] + [e["args"].get("text", "") for e in reversed(by_tool("note"))]:
        m = re.search(r"band\s*gap[^0-9\n]{0,25}([0-9]+\.[0-9]+)\s*eV", text, re.I)
        if m:
            reported = float(m.group(1)); break
    if gap is not None or reported is not None:
        plot_ok = [s["ok_step"] for p, s in scripts.items() if "plot" in _name(p).lower() and s["ok_step"]]
        txt = "band gap: " + (f"{gap:.3f} eV computed from the bands output" if gap is not None else "not computable from the outputs")
        if reported is not None:
            txt += f"; agent reported {reported:g} eV"
            if gap is not None and abs(reported - gap) > 0.05:
                txt += "  <-- DISAGREES"
        chosen.append(_Item(plot_ok[-1] if plot_ok else None, txt))

    lines += ["", "2. Results and key choices (in the order they were made)"]
    if not chosen:
        lines.append("    (none recorded -- the run did not get as far as writing QE inputs)")
    for it in sorted(chosen, key=lambda i: (i.step is None, i.step or 0)):
        lines += _wrap(it.text, 4, f"{_step_label(it.step)}  ")

    # -- 3. uncertainties and difficulties --------------------------------------
    lines += ["", "3. Uncertainties and difficulties"]
    trouble = False
    for path, s in scripts.items():
        if s["fails"]:
            trouble = True
            outcome = f"fixed at step {s['ok_step']}" if s["status"] == "ok" else "NOT fixed"
            lines += _wrap(f"{_name(path)}: {len(s['fails'])} failed run(s), {outcome}", 4, "- ")
            for step, err in s["fails"][:4]:
                lines += _wrap(f"step {step}: {err}", 8)
    for e in qe_runs:
        r = e.get("result") or {}
        if not (isinstance(r, dict) and r.get("success")):
            trouble = True
            lines += _wrap(f"Quantum ESPRESSO failed at step {e['step']}: {_last_line(r.get('stderr', '') if isinstance(r, dict) else r)}", 4, "- ")
    for path, s in scripts.items():
        if s.get("writes_since_run", 0) >= 3:
            trouble = True
            lines += _wrap(f"{_name(path)} was rewritten {s['writes_since_run']} times without being run in between "
                           "-- a rewrite loop", 4, "- ")
    if qe_runs:
        last_qe = qe_runs[-1]
        r = last_qe.get("result") or {}
        regenerated = [e["step"] for e in history if e["step"] > last_qe["step"] and e["tool"] == "run_python"
                       and run_succeeded(e.get("result")) and re.search(r"setup|job|input", _name(e["args"].get("path", "")), re.I)]
        if not (isinstance(r, dict) and r.get("success")) and regenerated:
            trouble = True
            lines += _wrap(f"the QE inputs were regenerated at step {regenerated[-1]} after the last QE run "
                           f"(step {last_qe['step']}, which failed), but QE was never run again -- any outputs on disk "
                           "are from the failed run", 4, "- ")
    clusters = _doc_clusters(history)
    if clusters:
        trouble = True
        lines.append("    - documentation lookups, grouped by what the agent did next:")
        for target, group in sorted(clusters, key=lambda g: -len(g[1])):
            queries = "; ".join(dict.fromkeys(str(g["args"].get("query", ""))[:70] for g in group))
            lines += _wrap(f"{target}: {len(group)} lookup(s) (step{'s' if len(group) > 1 else ''} "
                           f"{group[0]['step']}{'-' + str(group[-1]['step']) if len(group) > 1 else ''}): {queries}", 8)
    corners, labelled = kpath_info.get("corners"), {i for i, _ in kpath_info.get("labels", [])}
    if corners and labelled is not None and not set(corners) <= labelled:
        trouble = True
        missing = [i for i in corners if i not in labelled]
        lines += _wrap(f"k-path labels look incomplete: the path has {len(corners)} segment endpoints "
                       f"(k-point indices {', '.join(map(str, corners))}) but band_labels.dat names "
                       f"{len(labelled)}; unlabelled: {', '.join(map(str, missing))}. Check the band plot's tick labels.", 4, "- ")
    if refusals:
        trouble = True
        what = ", ".join(f"step {e['step']} ({e['args'].get('tool')})" for e in refusals[:8])
        lines += _wrap(f"harness refused {len(refusals)} call(s) as loops: {what}", 4, "- ")
    if parse_errors:
        trouble = True
        lines += _wrap(f"{len(parse_errors)} step(s) produced output that was not a valid tool call", 4, "- ")
    doubts = _doubt_sentences([(e["step"], e["args"].get("text", "")) for e in by_tool("note")])
    if doubts:
        trouble = True
        lines.append("    - notes where the agent flagged doubt or an assumption:")
        for step, sent in doubts:
            lines += _wrap(sent, 8, f"step {step}: ")
    sections = _summary_sections(final_result)
    own = {k: v for k, v in sections.items() if re.search(r"ASSUMPTION|REVIEW|CAVEAT|UNCERTAIN|LIMITATION", k)}
    if own:
        trouble = True
        for head, body in own.items():
            lines.append(f"    - agent's own {head.lower()}:")
            for item in [l.strip(" -*") for l in body.splitlines() if l.strip(" -*")][:8]:
                lines += _wrap(item, 8, "* ")
    elif final_result:
        caveats = _doubt_sentences([(None, final_result)], limit=4)
        if caveats:
            trouble = True
            lines.append("    - caveats in the agent's done summary:")
            for _, sent in caveats:
                lines += _wrap(sent, 8, "* ")
    if not completed:
        trouble = True
        failing = [f"{_name(p)} (since step {s['fails'][-1][0]}: {s['fails'][-1][1]})" for p, s in scripts.items() if s["status"] == "failing"]
        tail = [e["tool"] for e in history[-12:]]
        lines += _wrap("the run stopped without calling done"
                       + (f"; still failing: {'; '.join(failing)}" if failing else "")
                       + f"; its last 12 actions: {', '.join(tail)}", 4, "- ")
    if not trouble:
        lines.append("    nothing notable: no failed runs, doc lookups, refusals or flagged doubts")

    return "\n".join(lines) + "\n"


def load_log_entry(path: str | Path, index: int = -1) -> tuple[dict, Path]:
    """A run record from a log.jsonl (or a work dir holding one), and the
    work dir it ran in."""
    path = Path(path)
    log = path / "log.jsonl" if path.is_dir() else path
    lines = [l for l in log.read_text().splitlines() if l.strip()]
    if not lines:
        raise ValueError(f"{log} has no recorded runs (a run writes its record when it finishes)")
    if not -len(lines) <= index < len(lines):
        raise ValueError(f"{log} has {len(lines)} recorded run(s); index {index} is out of range")
    entry = json.loads(lines[index])
    work_dir = Path(entry.get("work_dir") or log.parent)
    if not work_dir.exists() and path.is_dir():
        work_dir = path
    return entry, work_dir


def summarize_log(path: str | Path, index: int = -1) -> str:
    """Summary of a recorded run: `path` is a log.jsonl or a work dir."""
    entry, work_dir = load_log_entry(path, index)
    fs = entry.get("final_state") or {}
    finished_at = None
    if entry.get("timestamp"):
        from datetime import datetime
        try:
            finished_at = datetime.fromisoformat(entry["timestamp"]).timestamp()
        except ValueError:
            pass
    return summarize_run(
        fs.get("history", []),
        work_dir,
        final_result=fs.get("final_result"),
        completed=bool(entry.get("completed")),
        runtime_seconds=entry.get("runtime"),
        model=entry.get("model"),
        finished_at=finished_at,
    )
