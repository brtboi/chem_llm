"""A readable summary of a finished agent run, built deterministically.

Everything comes from the run's own record -- the tool calls and their
results, the scripts it wrote (their logged contents), its notes and done
summary -- plus the QE inputs and outputs left in the work directory. No
model is involved, so nothing in the summary can be invented: a value is
either read from a file or a tool result, or it is reported as missing.

    from chem_llm.report import summarize_log
    print(summarize_log("sandbox/test29"))          # latest run logged there

Sections:
  1. What the agent did -- step by step: data pulled from the Materials
     Project and Pseudo-Dojo, scripts written and run, validation, QE runs.
  2. DFT setup -- for a physicist or chemist reproducing the calculation:
     structure, pseudopotentials, functional, spin-orbit, cutoffs, SCF
     settings, k-path, bands, the perturbed ensemble, QE version.
  3. Agent and harness -- for CS / LLM engineers: base model and decoding,
     prompt design, tools, context management, harness guards.
  4. Errors, uncertainties and difficulties -- failed attempts and their
     errors, validator findings, where documentation lookups clustered,
     refusals, notes that flag doubt, the agent's own caveats.
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


def _measured_perturbations(structures: Path, finished_at: float | None) -> str | None:
    """Per-species per-axis RMS displacement and RMS strain, measured from
    the structures the agent wrote (structure_000 = unperturbed base)."""
    base_file = structures / "structure_000.cif"
    others = sorted(structures.glob("structure_[0-9][0-9][0-9].cif"))[1:]
    if not base_file.exists() or not others:
        return None
    if finished_at is not None and any(f.stat().st_mtime > finished_at + 60 for f in [base_file, *others]):
        return None  # written by a later run
    try:
        import numpy as np
        from ase.io import read

        base = read(base_file)
        symbols = base.get_chemical_symbols()
        disp = {s: [] for s in dict.fromkeys(symbols)}
        strains = []
        for f in others:
            a = read(f)
            if len(a) != len(base):
                return None
            d = a.get_scaled_positions() - base.get_scaled_positions()
            d -= np.round(d)
            for s, v in zip(symbols, d @ a.cell[:]):
                disp[s].append(v)
            deform = np.linalg.solve(base.cell[:], a.cell[:]).T
            strains.append(0.5 * (deform + deform.T) - np.eye(3))
    except Exception:
        return None
    per = ", ".join(f"{s} {np.sqrt((np.array(v) ** 2).mean()):.3f}" for s, v in disp.items())
    strain = np.sqrt((np.array(strains) ** 2).mean()) * 100
    return f"measured over {len(others)} structures: per-axis RMS displacement {per} A; RMS strain {strain:.2f}%"


def _kpath(bands_in: str | None, labels_txt: str | None) -> dict:
    """The bands k-path: type, points, labels, discontinuities and segment
    endpoints (the same reading validate_qe_input's k-path check uses)."""
    from .kpath_check import _labels, read_kpoints, segment_endpoints

    info: dict = {}
    if bands_in:
        kind, pts = read_kpoints(bands_in)
        info["kind"] = kind
        if len(pts):
            info["n"], info["points"] = len(pts), pts
    if labels_txt:
        info["labels"] = sorted(_labels(labels_txt))
    pts = info.get("points")
    if (info.get("kind") or "") in ("crystal", "tpiba") and pts is not None and len(pts) > 2:
        info["corners"], info["breaks"] = segment_endpoints(pts)
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
    # markdown headings ("### Scientific Assumptions") -> "Scientific Assumptions:"
    text = re.sub(r"(?m)^\s*#{1,6}\s*(.+?)\s*:?\s*$", r"\1:", text)
    parts = re.split(r"(?m)^\s*([A-Z][A-Z0-9 /&()'-]{3,}|[A-Z][A-Za-z0-9 /&()'-]{3,60}(?=:\s*$)):", text)
    return {parts[i].strip().upper(): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


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

_RELATIVITY = {"sr": "scalar-relativistic", "fr": "fully relativistic"}


def _bullet(lines: list[str], text: str, indent: int = 4) -> None:
    lines += _wrap(text, indent, "- ")


def _pw_out_facts(text: str | None) -> dict:
    """Run facts Quantum ESPRESSO writes into pw.out."""
    if not text:
        return {}
    facts = {}
    for key, pattern in (
        ("version", r"Program PWSCF (v\.\S+)"),
        ("xc", r"Exchange-correlation\s*=\s*(\S+)"),
        ("mpi", r"Number of MPI processes:\s*(\d+)"),
        ("threads", r"Threads/MPI process:\s*(\d+)"),
        ("nk_irreducible", r"number of k points\s*=\s*(\d+)"),
        ("smearing", r"smearing,\s*width \(Ry\)\s*=\s*(\S+)"),
    ):
        m = re.search(pattern, text)
        if m:
            facts[key] = m.group(1)
    facts["gpu"] = bool(re.search(r"GPU acceleration is ACTIVE", text))
    facts["soc"] = bool(re.search(r"spin-orbit", text, re.I))
    return facts


def _structure_facts(calc: Path) -> dict:
    """Cell and atoms actually used, read back from pw.in."""
    try:
        from ase.io import read
        atoms = read(calc / "pw.in", format="espresso-in")
    except Exception:
        return {}
    lengths, angles = atoms.cell.cellpar()[:3], atoms.cell.cellpar()[3:]
    facts = {"formula": atoms.get_chemical_formula(), "n_atoms": len(atoms),
             "lengths": lengths, "angles": angles, "volume": atoms.get_volume()}
    try:
        import spglib
        ds = spglib.get_symmetry_dataset((atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers()), symprec=1e-3)
        if ds is not None:
            facts["spacegroup"] = f"{ds.international} (#{ds.number})"
    except Exception:
        pass
    if len(atoms) <= 12:
        facts["positions"] = [(a.symbol, tuple(f)) for a, f in zip(atoms, atoms.get_scaled_positions())]
    return facts


def _section_steps(history, work_dir, scripts, by_tool) -> list[str]:
    """1. What the agent did, in order."""
    did: list[_Item] = []
    for e in by_tool("generate_cif"):
        r = e.get("result") or {}
        sg = r.get("space_group", {}) if isinstance(r, dict) else {}
        if isinstance(r, dict) and r.get("success"):
            did.append(_Item(e["step"], f"fetched the {e['args'].get('composition')} structure "
                                        f"({sg.get('symbol', e['args'].get('spacegroup_symbol'))}) from the Materials Project: "
                                        f"{r.get('selected_material_id')} -> {_rel(r.get('output_path', ''), work_dir)}"))
        else:
            did.append(_Item(e["step"], f"Materials Project lookup FAILED: {_last_line(r.get('stderr', r) if isinstance(r, dict) else r)}"))
    for e in by_tool("get_pseudopotential"):
        a, r = e["args"], e.get("result") or {}
        what = f"{a.get('element')} pseudopotential from Pseudo-Dojo"
        if isinstance(r, dict) and r.get("success"):
            did.append(_Item(e["step"], f"fetched the {what} -> {_name(r.get('output_path', ''))}"))
        else:
            did.append(_Item(e["step"], f"fetching the {what} FAILED: {_last_line(r.get('stderr', '') if isinstance(r, dict) else r)}"))
    for path, s in scripts.items():
        runs = f"run {s['runs']}x" + (f" ({len(s['fails'])} failed)" if s["fails"] else "")
        end = (f"OK at step {s['ok_step']}" if s["status"] == "ok"
               else f"FAILING since step {s['fails'][-1][0]}" if s["status"] == "failing" else s["status"])
        versions = f"{s['writes']} version{'s' if s['writes'] != 1 else ''}"
        did.append(_Item(s["first"], f"wrote {_name(path)}: {versions}, {runs} -> {end}"))
    for e in by_tool("validate_qe_input"):
        r = e.get("result") or {}
        ok = isinstance(r, dict) and r.get("success")
        did.append(_Item(e["step"], f"validated {_rel(e['args'].get('path', ''), work_dir)}: "
                                    + ("OK" if ok else f"issues {r.get('issues') if isinstance(r, dict) else r}")))
    for e in by_tool("run_espresso_workflow"):
        r = e.get("result") or {}
        ok = isinstance(r, dict) and r.get("success")
        steps = ", ".join(r.get("completed_steps", [])) if isinstance(r, dict) else ""
        did.append(_Item(e["step"], f"ran Quantum ESPRESSO in {_rel(e['args'].get('directory', ''), work_dir)}: "
                                    + (f"OK ({steps})" if ok else f"FAILED -- {_last_line(r.get('stderr', '') if isinstance(r, dict) else r)}")))
    n_docs = len(by_tool("search_docs"))
    if n_docs:
        did.append(_Item(None, f"{n_docs} documentation lookup(s) along the way (where they clustered: section 4)"))
    lines = ["", "1. What the agent did"]
    for it in sorted(did, key=lambda i: (i.step is None, i.step or 0)):
        lines += _wrap(it.text, 4, f"{_step_label(it.step)}  ")
    return lines


def _section_setup(history, work_dir, calc, stale, found, scripts, by_tool, final_result, finished_at) -> list[str]:
    """2. The DFT setup, for a physicist or chemist reproducing the calculation."""
    lines = ["", "2. DFT setup (to reproduce the calculation)"]
    pw_in = None if stale else _read(calc / "pw.in")
    bands_in = None if stale else _read(calc / "bands.in")
    pw_out = None if stale else _read(calc / "pw.out")
    v, vb, qe = _namelist_values(pw_in or ""), _namelist_values(bands_in or ""), _pw_out_facts(pw_out)
    st = _structure_facts(calc) if pw_in else {}
    cifs, pseudos = by_tool("generate_cif"), by_tool("get_pseudopotential")

    if not (pw_in or pseudos or cifs):
        lines.append("    (not recorded -- the run did not get as far as fetching inputs)")
        return lines
    if stale:
        lines.append("    (QE inputs and outputs in this work dir were overwritten by a later run; pw.x settings omitted)")

    lines.append("    Structure")
    for e in cifs:
        r = e.get("result") or {}
        if isinstance(r, dict) and r.get("success"):
            sg = r.get("space_group") or {}
            src = f"Materials Project {r.get('selected_material_id')} ({e['args'].get('composition')}, {sg.get('symbol', '?')}"
            src += f" #{sg['number']})" if sg.get("number") else ")"
            _bullet(lines, f"source: {src}", 6)
    if st:
        a, b, c = st["lengths"]
        al, be, ga = st["angles"]
        _bullet(lines, f"cell used for DFT: {st['formula']}, {st['n_atoms']} atoms"
                       + (f", {st['spacegroup']}" if st.get("spacegroup") else "")
                       + f"; a, b, c = {a:.4f}, {b:.4f}, {c:.4f} A; alpha, beta, gamma = {al:.2f}, {be:.2f}, {ga:.2f} deg;"
                       f" volume {st['volume']:.3f} A^3", 6)
        if st.get("positions"):
            pos = "; ".join(f"{sym} ({', '.join(f'{x:.4f}' for x in f)})" for sym, f in st["positions"])
            _bullet(lines, f"fractional positions: {pos}", 6)

    if pseudos:
        lines.append("    Pseudopotentials (Pseudo-Dojo)")
        for e in pseudos:
            a, r = e["args"], e.get("result") or {}
            if not (isinstance(r, dict) and r.get("success")):
                continue
            f = r.get("file") or _upf_facts(r.get("output_path"))
            rec = r.get("recommended_ecutwfc") or {}
            kind = {"NC": "norm-conserving", "US": "ultrasoft", "USPP": "ultrasoft", "PAW": "PAW"}.get(
                (f.get("pseudo_type") or "").upper(), {"nc": "norm-conserving", "paw": "PAW"}.get(a.get("kind"), a.get("kind", "?")))
            rel = {"scalar": "scalar-relativistic", "full": "fully relativistic"}.get(f.get("relativistic"), _RELATIVITY.get(a.get("relativity"), "?"))
            parts = [kind, rel, (f.get("functional") or a.get("generator") or "?"), f"{a.get('accuracy', '?')} table"]
            z = f"z_valence {f['z_valence']:g}" if f.get("z_valence") is not None else "z_valence unknown"
            cut = f"recommended ecutwfc {rec['value']} Ha ({2 * rec['value']:g} Ry)" if rec.get("value") else "no cutoff hint recorded"
            _bullet(lines, f"{a.get('element')}: {_name(r.get('output_path', ''))} -- {', '.join(parts)}; {z}; {cut}", 6)

    if pw_in:
        lines.append("    pw.x settings")
        rels = {e["args"].get("relativity") for e in pseudos}
        soc_on = v.get("lspinorb", "").lower().strip(".") in ("true", "t")
        _bullet(lines, f"exchange-correlation: {qe.get('xc') or ', '.join(sorted({(e.get('result') or {}).get('file', {}).get('functional', '?') for e in pseudos})) or '?'}"
                       " (taken from the pseudopotentials)", 6)
        _bullet(lines, "spin-orbit coupling: " + ("ON (noncolin = .true., lspinorb = .true.)" if soc_on
                       else "off" + (" -- but fully relativistic pseudopotentials were fetched" if "fr" in rels else "")), 6)
        recs = [((e.get("result") or {}).get("recommended_ecutwfc") or {}).get("value") for e in pseudos]
        recs = [x for x in recs if isinstance(x, (int, float))]
        _bullet(lines, f"cutoffs: ecutwfc = {v.get('ecutwfc', '?')} Ry, ecutrho = {v.get('ecutrho', '?')} Ry"
                       + (f" (largest recommended: {max(recs):g} Ha = {2 * max(recs):g} Ry)" if recs else ""), 6)
        occ = f"occupations = {v.get('occupations', '?')}" + (f", smearing = {v['smearing']}, degauss = {v['degauss']} Ry"
                                                              if "smearing" in v else "")
        if "nspin" in v:
            occ += f", nspin = {v['nspin']}"
        _bullet(lines, occ, 6)
        mesh = re.search(r"K_POINTS\s*\{?\s*automatic\s*\}?\s*\n\s*([\d ]+)", pw_in, re.I)
        if mesh:
            m = mesh.group(1).split()
            _bullet(lines, f"SCF k-mesh: {' x '.join(m[:3])} Monkhorst-Pack, shift {' '.join(m[3:]) or '0 0 0'}"
                           + (f" ({qe['nk_irreducible']} irreducible k-points)" if qe.get("nk_irreducible") else ""), 6)
        _bullet(lines, f"SCF convergence: conv_thr = {v.get('conv_thr', '?')} Ry", 6)

    kp = _kpath(bands_in, None) if bands_in else {}
    if bands_in:
        from .kpath_check import path_string
        lines.append("    Band structure")
        path = path_string(calc / "bands.in")
        _bullet(lines, f"k-path: {path or '?'} ({kp.get('n', '?')} k-points, K_POINTS {kp.get('kind', '?')}; "
                       "points named from their coordinates by symmetry)", 6)
        _bullet(lines, f"nbnd = {vb.get('nbnd', '?')} in the bands calculation", 6)
    if found.get("n_electrons") or found.get("band_gap") is not None:
        n_el = found.get("n_electrons")
        soc = qe.get("soc")
        occ_txt = (f"{n_el:g} electrons -> {int(round(n_el if soc else n_el / 2))} occupied bands"
                   f" ({'one electron per band, spin-orbit' if soc else 'two electrons per band'})") if n_el else ""
        if occ_txt:
            _bullet(lines, occ_txt, 6)
        gap = found.get("band_gap")
        if gap is not None:
            _bullet(lines, f"band gap: {gap:.3f} eV (computed here from the bands output; DFT band gaps are typically underestimated)", 6)
        _bullet(lines, "SCF " + ("converged" if found.get("scf_converged") else "NOT converged")
                       + "; " + ("all three QE stages (SCF, bands, bands.x) finished" if found.get("dft_finished") else "QE stages incomplete"), 6)

    measured = _measured_perturbations(work_dir / "structures", finished_at)
    gen = _structure_script(scripts)
    if measured or gen:
        lines.append("    Perturbed-structure ensemble")
        content = gen[1]["content"] if gen else ""
        temp = re.search(r"^\s*(?:T|TEMP\w*|temp\w*|T_K|temperature\w*)\s*=\s*([\d.]+)", content or "", re.M)
        if temp:
            _bullet(lines, f"temperature: {temp.group(1)} K (as set in {_name(gen[0])})", 6)
        if measured:
            _bullet(lines, measured, 6)
        n = len(list((work_dir / "structures").glob("structure_[0-9][0-9][0-9].cif")))
        if n:
            _bullet(lines, f"{n} structures; DFT was run on structure_000 (unperturbed) only", 6)

    if qe:
        lines.append("    Software")
        par = f"{qe.get('mpi', '?')} MPI process(es) x {qe.get('threads', '?')} thread(s), GPU {'on' if qe.get('gpu') else 'off'}"
        _bullet(lines, f"Quantum ESPRESSO {qe.get('version', '?')} (pw.x, bands.x); {par}", 6)
    return lines


def _section_agent(history, config, model, completed, runtime_seconds, by_tool) -> list[str]:
    """3. How the agent and harness ran, for CS / LLM engineers."""
    from collections import Counter

    lines = ["", "3. Agent and harness (for CS / LLM engineers)"]
    cfg = config or {}
    backend = cfg.get("backend") or ("qwen" if model and "qwen" in model.lower() else "claude" if model and "claude" in model.lower() else None)
    where = {"qwen": "local open-weights model via transformers, bf16, sharded across the node's GPUs",
             "claude": "Anthropic API"}.get(backend, "")
    _bullet(lines, f"base model: {model or cfg.get('model', '?')}" + (f" ({where})" if where else ""))
    if cfg.get("decoding"):
        _bullet(lines, "decoding: " + ", ".join(f"{k} {v}" for k, v in cfg["decoding"].items())
                       + ("; greedy decoding is deterministic, so a rerun of the same prompt repeats the same trajectory"
                          if cfg["decoding"].get("do_sample") is False else ""))
    calls = Counter(e["tool"] for e in history)
    actions = sum(n for t, n in calls.items() if t not in ("refused", "parse_error", "api_error"))
    _bullet(lines, f"run: {'completed' if completed else 'did NOT complete'} in {len(history)} steps"
                   + (f" of {cfg['max_steps']} allowed" if cfg.get("max_steps") else "")
                   + (f", {runtime_seconds / 60:.1f} min" if runtime_seconds else "")
                   + f"; {actions} tool calls: " + ", ".join(f"{t} {n}" for t, n in calls.most_common()
                                                             if t not in ("refused", "parse_error", "api_error")))
    _bullet(lines, "prompt: a fixed system prompt"
                   + (f" ({cfg['system_prompt_chars']:,} chars)" if cfg.get("system_prompt_chars") else "")
                   + " -- agent protocol (one JSON tool call per step), numbered working rules, INVARIANTS the agent's"
                   " scripts must assert in code (atom/species counts and electron count read from the actual"
                   " structure and pw.out, never from memory; k-path from seekpath on the cell in use; unit"
                   " conversions), and DECISION RULES (relativistic treatment, k-path source, primitive cell, plotting)"
                   " -- plus a task prompt"
                   + (f" ({cfg['task_chars']:,} chars)" if cfg.get("task_chars") else "")
                   + " giving a 7-step workflow (fetch structure, perturb, pseudopotentials, decide settings, write"
                   " inputs, run QE, plot) with done criteria. No examples or templates: the agent writes every script.")
    _bullet(lines, "tools: deterministic Python functions, each returning JSON -- Materials Project fetch, Pseudo-Dojo fetch"
                   " (reports z_valence/relativity/functional parsed from the downloaded UPF and the matching recommended"
                   " cutoff), a k-path tool (seekpath's standard path, shared by the QE input and the plot), file"
                   " read/write, Python execution (exit code), a QE input validator (namelists, nat/ntyp, k-path),"
                   " a QE runner, and a documentation search (BM25 + embeddings + reranker"
                   " over the ASE and QE docs). Facts the model is weak at come from tools, not from its memory.")
    ctx = f"last {cfg['max_history']} tool calls" if cfg.get("max_history") else "most recent tool calls"
    trim = cfg.get("trim_history")
    _bullet(lines, f"context: each step re-prompts with the task, notes, files written, unresolved errors and the {ctx}"
                   + ("; trim_history on (superseded file versions, resolved failures and duplicate notes dropped, plus"
                      " a progress ledger)" if trim else "; trim_history off" if trim is False else ""))
    _bullet(lines, "harness guards: a second consecutive note, a repeated doc search, or rewriting a file with identical"
                   " content is refused; model output that is"
                   " not a valid tool call is fed back as an error; results (gap, convergence, outputs) are re-derived"
                   " from files on disk rather than trusted from the agent's own summary")
    issues = [f"{calls['parse_error']} unparseable output(s)" if calls.get("parse_error") else "",
              f"{calls['refused']} refused call(s)" if calls.get("refused") else "",
              f"{calls['api_error']} API error(s)" if calls.get("api_error") else ""]
    if any(issues):
        _bullet(lines, "protocol issues this run: " + ", ".join(i for i in issues if i))
    if cfg.get("qe_command"):
        _bullet(lines, f"QE launch command: {cfg['qe_command']}")
    if not config:
        _bullet(lines, "(this run predates configuration logging; decoding and history settings were not recorded)")
    return lines


def _section_difficulties(history, work_dir, scripts, by_tool, final_result, completed, kpath_info) -> list[str]:
    """4. Errors, uncertainties and difficulties."""
    lines = ["", "4. Errors, uncertainties and difficulties"]
    qe_runs, refusals, parse_errors = by_tool("run_espresso_workflow"), by_tool("refused"), by_tool("parse_error")
    trouble = False
    for path, s in scripts.items():
        if s["fails"]:
            trouble = True
            outcome = f"fixed at step {s['ok_step']}" if s["status"] == "ok" else "NOT fixed"
            _bullet(lines, f"{_name(path)}: {len(s['fails'])} failed run(s), {outcome}")
            for step, err in s["fails"][:4]:
                lines += _wrap(f"step {step}: {err}", 8)
    for e in qe_runs:
        r = e.get("result") or {}
        if not (isinstance(r, dict) and r.get("success")):
            trouble = True
            _bullet(lines, f"Quantum ESPRESSO failed at step {e['step']}: {_last_line(r.get('stderr', '') if isinstance(r, dict) else r)}")
    for e in by_tool("validate_qe_input"):
        r = e.get("result") or {}
        if isinstance(r, dict) and not r.get("success") and r.get("issues"):
            trouble = True
            _bullet(lines, f"validator flagged {_rel(e['args'].get('path', ''), work_dir)} at step {e['step']}: {r['issues'][0]}")
    for path, s in scripts.items():
        if s.get("writes_since_run", 0) >= 3:
            trouble = True
            _bullet(lines, f"{_name(path)} was rewritten {s['writes_since_run']} times without being run in between -- a rewrite loop")
    if qe_runs:
        last_qe = qe_runs[-1]
        r = last_qe.get("result") or {}
        regenerated = [e["step"] for e in history if e["step"] > last_qe["step"] and e["tool"] == "run_python"
                       and run_succeeded(e.get("result")) and re.search(r"setup|job|input", _name(e["args"].get("path", "")), re.I)]
        if not (isinstance(r, dict) and r.get("success")) and regenerated:
            trouble = True
            _bullet(lines, f"the QE inputs were regenerated at step {regenerated[-1]} after the last QE run (step "
                           f"{last_qe['step']}, which failed), but QE was never run again -- any outputs on disk are from the failed run")
    clusters = _doc_clusters(history)
    if clusters:
        trouble = True
        lines.append("    - documentation lookups, grouped by what the agent did next:")
        for target, group in sorted(clusters, key=lambda g: -len(g[1])):
            queries = "; ".join(dict.fromkeys(str(g["args"].get("query", ""))[:70] for g in group))
            lines += _wrap(f"{target}: {len(group)} lookup(s) (step{'s' if len(group) > 1 else ''} "
                           f"{group[0]['step']}{'-' + str(group[-1]['step']) if len(group) > 1 else ''}): {queries}", 8)
    from .kpath_check import unlabelled_endpoints

    corners, labelled = kpath_info.get("corners"), {i for i, _ in kpath_info.get("labels", [])}
    missing = (unlabelled_endpoints(kpath_info["points"], corners, labelled)
               if corners and kpath_info.get("labels") else [])
    if missing and unlabelled_endpoints(kpath_info["points"], corners, labelled, offset=1):
        trouble = True
        _bullet(lines, f"k-path labels look incomplete: the path has {len(corners)} segment endpoints but band_labels.dat "
                       f"names {len(labelled)}; unlabelled k-point indices: {', '.join(map(str, missing))}. Check the plot's tick labels.")
    if refusals:
        trouble = True
        what = ", ".join(f"step {e['step']} ({e['args'].get('tool')})" for e in refusals[:8])
        _bullet(lines, f"harness refused {len(refusals)} call(s) as loops: {what}")
    if parse_errors:
        trouble = True
        _bullet(lines, f"{len(parse_errors)} step(s) produced output that was not a valid tool call")
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
            for item in [l.replace("**", "").strip(" -*") for l in body.splitlines() if l.strip(" -*")][:8]:
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
        _bullet(lines, "the run stopped without calling done"
                       + (f"; still failing: {'; '.join(failing)}" if failing else "")
                       + f"; its last 12 actions: {', '.join(tail)}")
    if not trouble:
        lines.append("    nothing notable: no failed runs, doc lookups, refusals or flagged doubts")
    return lines


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
    config: dict | None = None,
) -> str:
    """Plain-text summary of one run. `history` is AgentState.history.

    finished_at: the run's end time (epoch seconds). When given, files in
    the work dir modified after it belong to a later run and are ignored.
    config: DFTAgent.run_config() for the run, if recorded.
    """
    from .pipeline import _inspect_outputs

    work_dir = Path(work_dir)
    scripts = _scripts(history)
    by_tool = lambda t: [e for e in history if e["tool"] == t]
    cifs, qe_runs = by_tool("generate_cif"), by_tool("run_espresso_workflow")

    compound = next((e["args"].get("composition") for e in cifs), None) or compound
    spacegroup = next((e["args"].get("spacegroup_symbol") for e in cifs), None) or spacegroup

    # the calculation directory the QE workflow ran in (default calculations/000)
    calc_rel = next((e["args"].get("directory") for e in reversed(qe_runs) if e["args"].get("directory")), "calculations/000")
    calc = Path(calc_rel) if Path(calc_rel).is_absolute() else work_dir / calc_rel
    # QE inputs/outputs modified after the run finished were written by a
    # later run (plots and other files the user adds afterwards don't count)
    qe_files = [f for f in calc.glob("*") if f.is_file()
                and re.search(r"\.(in|out|xml)$|\.bands\.dat", f.name)]
    stale = finished_at is not None and any(f.stat().st_mtime > finished_at + 60 for f in qe_files)
    found = _inspect_outputs(work_dir, job=calc.name) if calc.parent.name == "calculations" and not stale else {}
    kpath_info = {} if stale else _kpath(_read(calc / "bands.in"), _read(calc / "band_labels.dat"))

    title = f"chem_llm run summary -- {compound or 'unknown compound'}" + (f" ({spacegroup})" if spacegroup else "")
    lines = [title, "=" * len(title)]
    lines.append(f"{'completed' if completed else 'DID NOT COMPLETE'} in {len(history)} steps"
                 + (f", {runtime_seconds / 60:.1f} min" if runtime_seconds else "") + (f" | {model}" if model else ""))
    lines.append(f"work dir: {work_dir}")
    if found.get("band_gap") is not None:
        lines.append(f"band gap: {found['band_gap']:.3f} eV (computed from the bands output)")
    if found.get("plot"):
        lines.append(f"plot:     {_rel(found['plot'], work_dir)}")
    if stale:
        lines += _wrap(f"the files in {_rel(calc, work_dir)} were written by a later run in this work dir, so "
                       "values read from QE inputs/outputs are omitted below", 0, "note:     ")

    lines += _section_steps(history, work_dir, scripts, by_tool)
    lines += _section_setup(history, work_dir, calc, stale, found, scripts, by_tool, final_result, finished_at)
    lines += _section_agent(history, config, model, completed, runtime_seconds, by_tool)
    lines += _section_difficulties(history, work_dir, scripts, by_tool, final_result, completed, kpath_info)
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
        config=entry.get("config"),
    )
