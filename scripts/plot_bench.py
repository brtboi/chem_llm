#!/usr/bin/env python3
"""Charts of where the agent's time and tokens go, from the benchmark records.

    python scripts/plot_bench.py                 # latest sweep vs last week's test34 runs
    python scripts/plot_bench.py --sweep "sweep 8"

Writes PDFs and a README to bench/visualizations/. Reads bench/index.jsonl and
each run's record.json (format 2, chem_llm/runlog.py).

Last week's runs (test34 rerun, 2026-10-07, in-process transformers) predate
per-step timing and token logging, so their numbers are reconstructed and
cached in bench/visualizations/data/last_week.json:
- tokens: each step's prompt is rebuilt by replaying the run's recorded tool
  calls through AgentState, with the exact system prompt of the commit the run
  used, and counted with the Qwen tokenizer (validated on two sweep-8 runs
  against their logged tokens: 99.9%);
- time: the progress log stamps each tool call's start, so the gap between two
  stamps is this tool + the next model call; model time per call is fitted
  as a + b*prompt + c*completion on gaps after instant tools (notes, file
  writes), and the rest of each gap is tool time (validated on all 21 sweep-8
  runs: model time within 6% median, 14% worst; tool time within ~1 min).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from chem_llm.archive import read_index  # noqa: E402
from chem_llm.runlog import history_of, upgrade  # noqa: E402
from chem_llm.telemetry import OPERATION_LABELS, PHASE_LABELS, PHASES, run_metrics  # noqa: E402

BENCH = REPO / "bench"
OUT = BENCH / "visualizations"
LAST_WEEK_LABEL = "test34 rerun"
LAST_WEEK_COMMIT = "8030a6a"   # the commit the 2026-10-07 test34 rerun ran (its system prompt: 29,139 chars)

# --- style: categorical slots in fixed order (dataviz reference palette, light mode)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OPS = ("llm", "tools", "qe", "code")
OP_COLOR = dict(zip(OPS, SLOTS))
PHASE_COLOR = dict(zip(PHASES, SLOTS))


def style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8,
        "pdf.fonttype": 42,
    })


# ----------------------------------------------------------------------------
# data
def runs_of(label_prefix: str) -> list[tuple[dict, dict]]:
    """(index entry, record) for every run whose label starts with label_prefix."""
    out = []
    for e in read_index(BENCH):
        if (e.get("label") or "").startswith(label_prefix):
            out.append((e, upgrade(json.loads((BENCH / e["dir"] / "record.json").read_text()))))
    return out


def passing_runs(sweep: str, runs) -> list[tuple[dict, dict]]:
    """`runs` with each failed case replaced by its passing "<sweep> follow-up" rerun, if any.
    A failed run can be cheap for the wrong reason (sweep 8's CsPbBr3 skipped spin-orbit),
    so the comparison with last week uses the run that got the physics right."""
    fixed = {e["case"]: (e, r) for e, r in runs_of(sweep + " follow-up") if e.get("passed")}
    return [fixed.get(e["case"], (e, r)) if e.get("passed") is False else (e, r) for e, r in runs]


def latest_sweep() -> str:
    labels = [e.get("label") or "" for e in read_index(BENCH)]
    sweeps = sorted({m.group(0) for lab in labels if (m := re.match(r"sweep \d+", lab))},
                    key=lambda s: int(s.split()[1]))
    return sweeps[-1]


def _old_system_prompt(commit: str) -> str:
    """The system prompt as the package at `commit` built it."""
    with tempfile.TemporaryDirectory() as tmp:
        archive = subprocess.run(["git", "-C", str(REPO), "archive", commit, "chem_llm"], capture_output=True, check=True)
        subprocess.run(["tar", "-x", "-C", tmp], input=archive.stdout, check=True)
        code = "import sys; sys.path.insert(0, '.'); import chem_llm.agent_core as a; sys.stdout.write(a.SYSTEM_PROMPT_TEMPLATE)"
        return subprocess.run([sys.executable, "-c", code], cwd=tmp, capture_output=True, text=True, check=True).stdout


def _reconstruct_tokens(record: dict, system: str) -> list[tuple[int, int]]:
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer

    from chem_llm.settings import Settings
    from chem_llm.state import AgentState

    s = Settings.discover(REPO)
    tok = AutoTokenizer.from_pretrained(record.get("model") or "Qwen/Qwen3.8-27B", cache_dir=s.hf_cache_dir)
    cfg = record.get("config") or {}
    state, out = AgentState(record["prompt"]["task"]), []
    for i, e in enumerate(history_of(record), 1):
        user = state.context_summary(max_history=cfg.get("max_history", 32), trim=cfg.get("trim_history", True))
        text = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        args = dict(e.get("args") or {})
        if e["tool"] in ("parse_error", "api_error"):
            reply = str(args.get("raw_output", ""))
        elif e["tool"] == "refused":
            reply = json.dumps({"tool": args.pop("tool", ""), "args": args})
        else:
            reply = json.dumps({"tool": e["tool"], "args": args})
        out.append((len(tok(text, add_special_tokens=False)["input_ids"]), len(tok(reply)["input_ids"])))
        if e["tool"] == "note":
            state.add_note(args.get("text", ""))
        if e["tool"] == "parse_error":
            state.add_note(f"Step {i}: failed to parse model output as JSON: {e.get('result')}")
        state.log(e["tool"], e.get("args") or {}, e.get("result"))
    return out


INSTANT = {"note", "write_file", "read_file", "make_dir", "get_kpath", "validate_qe_input", "refused", "done",
           "parse_error"}


def _estimate_times(progress: str, tools: list[str], tokens: list[tuple[int, int]], runtime: float):
    stamps = {}
    for m in re.finditer(r"^\[\s*(\d+) \| (?:(\d+):)?(\d+):(\d+)\]", progress, re.M):
        stamps.setdefault(int(m.group(1)), int(m.group(2) or 0) * 3600 + int(m.group(3)) * 60 + int(m.group(4)))
    n = len(tools)
    t = []
    for i in range(n):
        t.append(stamps.get(i + 1, t[-1] if t else 0.0))
    gaps = [(t[i + 1] if i + 1 < n else runtime) - t[i] for i in range(n)]
    X = np.array([[1.0, *tokens[i + 1]] for i in range(n - 1) if tools[i] in INSTANT])
    y = np.array([gaps[i] for i in range(n - 1) if tools[i] in INSTANT])
    coef = np.maximum(np.linalg.lstsq(X, y, rcond=None)[0], 0)
    pred = [float(coef @ [1.0, *tk]) for tk in tokens]
    llm, tool = [float(t[0])] + [0.0] * (n - 1), [0.0] * n
    for i in range(n):
        tool[i] = 0.0 if tools[i] in INSTANT else max(0.0, gaps[i] - (pred[i + 1] if i + 1 < n else 0.0))
        if i + 1 < n:
            llm[i + 1] = gaps[i] - tool[i]
    return llm, tool


def last_week(refresh: bool = False) -> list[dict]:
    """Reconstructed metrics of last week's test34 runs (cached)."""
    cache = OUT / "data" / "last_week.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    system = _old_system_prompt(LAST_WEEK_COMMIT)
    out = []
    for e, rec in runs_of(LAST_WEEK_LABEL):
        history = history_of(rec)
        tokens = _reconstruct_tokens(rec, system)
        progress = (BENCH / e["dir"] / "progress.txt").read_text()
        llm, tool = _estimate_times(progress, [h["tool"] for h in history], tokens, rec["runtime"])
        timing = [{"step": i + 1, "tool": h["tool"], "llm_s": llm[i], "tool_s": tool[i],
                   "prompt_tokens": tokens[i][0], "completion_tokens": tokens[i][1], "history": [h["step"]]}
                  for i, h in enumerate(history)]
        state = SimpleNamespace(history=history, steps=timing, task=rec["prompt"]["task"])
        out.append({"case": e["case"], "runtime": rec["runtime"], "total_steps": rec["total_steps"],
                    "metrics": run_metrics(state, rec["runtime"]),
                    "prompt_tokens_per_step": [tk[0] for tk in tokens]})
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=1))
    return out


def steps_table() -> tuple[list[str], list[str], dict]:
    """{(case, sweep): (steps, passed)} over every labelled sweep."""
    table, sweeps = {}, []
    for e in read_index(BENCH):
        label = e.get("label") or ""
        m = re.match(r"(sweep \d+)(?!\s*follow)", label)
        sweep = "last week" if label.startswith(LAST_WEEK_LABEL) else (m.group(1) if m else None)
        if sweep is None:
            continue
        if sweep not in sweeps:
            sweeps.append(sweep)
        table[(e["case"], sweep)] = (e.get("steps"), e.get("passed"))
    cases = list(dict.fromkeys(e["case"] for e in read_index(BENCH) if e.get("case")))
    return cases, sweeps, table


# ----------------------------------------------------------------------------
# charts
def _stacked_barh(ax, labels, parts: dict[str, list[float]], colors: dict, scale=1.0):
    y = np.arange(len(labels))
    left = np.zeros(len(labels))
    for key, values in parts.items():
        v = np.asarray(values, float) * scale
        ax.barh(y, v, left=left, height=0.62, color=colors[key], edgecolor=SURFACE, linewidth=1.2)
        left += v
    ax.set_yticks(y, labels)
    if not ax.yaxis_inverted():  # panels share the y axis: invert it once
        ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    return left


def chart_operations(sweep: str, runs, path: Path):
    import matplotlib.pyplot as plt

    runs = sorted(runs, key=lambda er: -er[1]["runtime"])
    labels = [f"{e['case']}" + ("  (failed)" if e.get("passed") is False else "") for e, _ in runs]
    ops = [r["metrics"]["operations"] for _, r in runs]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11, 0.32 * len(runs) + 1.6), sharey=True,
                                     gridspec_kw={"width_ratios": [1.2, 1, 1]})
    totals = _stacked_barh(a1, labels, {k: [o[k]["time"] for o in ops] for k in OPS}, OP_COLOR, scale=1 / 60)
    for yi, tot in enumerate(totals):
        a1.text(tot + 0.2, yi, f"{tot:.1f}", va="center", fontsize=7, color=INK2)
    a1.set_xlabel("minutes")
    a1.set_title("Wall time by operation")
    _stacked_barh(a2, labels, {k: [100 * o[k]["time"] / max(o["total"]["time"], 1e-9) for o in ops] for k in OPS},
                  OP_COLOR)
    a2.set_xlim(0, 100)
    a2.set_xlabel("% of wall time")
    a2.set_title("Share of wall time")
    toks = _stacked_barh(a3, labels, {k: [o[k]["tokens"] for o in ops] for k in ("tools", "qe", "code")},
                         OP_COLOR, scale=1e-6)
    for yi, tot in enumerate(toks):
        a3.text(tot + 0.02, yi, f"{tot:.2f}", va="center", fontsize=7, color=INK2)
    a3.set_xlabel("million tokens")
    a3.set_title("Tokens, by the call they chose")
    handles = [plt.Rectangle((0, 0), 1, 1, color=OP_COLOR[k]) for k in OPS]
    fig.legend(handles, [OPERATION_LABELS[k] for k in OPS], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{sweep}: where each run's time and tokens go", x=0.01, ha="left", y=1.06, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_phases(sweep: str, runs, path: Path):
    import matplotlib.pyplot as plt

    runs = sorted(runs, key=lambda er: -er[1]["runtime"])
    labels = [e["case"] for e, _ in runs]
    ph = [r["metrics"]["phases"] for _, r in runs]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 0.32 * len(runs) + 1.8), sharey=True)
    _stacked_barh(a1, labels, {k: [p[k]["time"] for p in ph] for k in PHASES}, PHASE_COLOR, scale=1 / 60)
    a1.set_xlabel("minutes")
    a1.set_title("Time by workflow step")
    _stacked_barh(a2, labels, {k: [p[k]["tokens"] for p in ph] for k in PHASES}, PHASE_COLOR, scale=1e-6)
    a2.set_xlabel("million tokens")
    a2.set_title("Tokens by workflow step")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PHASE_COLOR[k]) for k in PHASES]
    fig.legend(handles, [PHASE_LABELS[k] for k in PHASES], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{sweep}: the seven workflow steps of the task", x=0.01, ha="left", y=1.1, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_vs_last_week(sweep: str, runs, old: list[dict], path: Path):
    import matplotlib.pyplot as plt

    now = {e["case"]: r for e, r in runs}
    cases = [o["case"] for o in old if o["case"] in now]
    oldm = {o["case"]: o for o in old}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    a1, a2, a3 = axes
    x = np.arange(len(cases))
    w = 0.38
    for off, src, hatch, name in ((-w / 2 - 0.01, "old", "////", "last week (in-process transformers, estimated split)"),
                                  (w / 2 + 0.01, "now", None, f"{sweep} (vLLM, logged)")):
        bottom = np.zeros(len(cases))
        for k in OPS:
            v = np.array([(oldm[c]["metrics"] if src == "old" else now[c]["metrics"])["operations"][k]["time"] / 60
                          for c in cases])
            a1.bar(x + off, v, w, bottom=bottom, color=OP_COLOR[k], edgecolor=SURFACE, linewidth=1.0,
                   hatch=hatch)
            bottom += v
        for xi, b in zip(x + off, bottom):
            a1.text(xi, b + 0.8, f"{b:.0f}", ha="center", fontsize=7, color=INK2)
    a1.set_xticks(x, cases)
    a1.set_ylabel("minutes")
    a1.set_title("Wall time: last week (hatched) vs now")
    a1.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=OP_COLOR[k]) for k in OPS]
    a1.legend(handles, [OPERATION_LABELS[k] for k in OPS], loc="upper right", fontsize=7)

    for ax, key, title, scale, unit in ((a2, "total_steps", "Agent steps", 1, ""),
                                        (a3, "tokens", "Total tokens (last week reconstructed)", 1e-6, "M")):
        o = [oldm[c]["total_steps"] if key == "total_steps" else oldm[c]["metrics"]["operations"]["total"]["tokens"]
             for c in cases]
        n = [now[c]["total_steps"] if key == "total_steps" else now[c]["total_tokens"] for c in cases]
        ax.bar(x - w / 2 - 0.01, np.array(o) * scale, w, color=SLOTS[0], hatch="////", edgecolor=SURFACE,
               label="last week")
        ax.bar(x + w / 2 + 0.01, np.array(n) * scale, w, color=SLOTS[0], edgecolor=SURFACE, label=sweep)
        ax.set_ylim(0, max(max(o), max(n)) * scale * 1.15)
        for xi, a, b in zip(x, o, n):
            ax.text(xi, max(a, b) * scale * 1.03, f"{(b - a) / a:+.0%}", ha="center", fontsize=7, color=INK2)
        ax.set_xticks(x, cases)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        if unit:
            ax.set_ylabel("million tokens")
    fig.legend([plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK2, hatch="////"),
                plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK2)],
               ["last week (left bar of each pair)", f"{sweep} (right bar)"], loc="upper right",
               ncol=2, bbox_to_anchor=(1.0, 1.04))
    fig.suptitle(f"The same five compounds: last week (test34 rerun, 2026-10-07) vs {sweep}",
                 x=0.01, ha="left", y=1.03, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_steps_history(path: Path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cases, sweeps, table = steps_table()
    cases = [c for c in cases if any((c, s) in table for s in sweeps)]
    grid = np.full((len(cases), len(sweeps)), np.nan)
    for i, c in enumerate(cases):
        for j, s in enumerate(sweeps):
            if (c, s) in table and table[(c, s)][0] is not None:
                grid[i, j] = table[(c, s)][0]
    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    cmap.set_bad(SURFACE)
    fig, ax = plt.subplots(figsize=(1.0 * len(sweeps) + 2.5, 0.3 * len(cases) + 1.6))
    im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    for i, c in enumerate(cases):
        for j, s in enumerate(sweeps):
            if (c, s) not in table or table[(c, s)][0] is None:
                continue
            steps, passed = table[(c, s)]
            dark = steps > np.nanmin(grid) + 0.55 * (np.nanmax(grid) - np.nanmin(grid))
            ax.text(j, i, f"{steps}" + (" ✗" if passed is False else ""), ha="center", va="center", fontsize=7,
                    color="white" if dark else INK)
    ax.set_xticks(range(len(sweeps)), sweeps, rotation=30, ha="right")
    ax.set_yticks(range(len(cases)), cases)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("agent steps", color=INK2)
    ax.set_title("Agent steps per run, every sweep (✗ = failed a check; blank = not run)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chart_prompt_growth(sweep: str, runs, old: list[dict], path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for _, r in runs:
        y = [s["tokens"] for s in r["steps"] if s.get("tokens") is not None]
        ax.plot(range(1, len(y) + 1), np.array(y) / 1e3, color=SLOTS[0], lw=0.8, alpha=0.35)
    for o in old:
        y = o["prompt_tokens_per_step"]
        ax.plot(range(1, len(y) + 1), np.array(y) / 1e3, color=SLOTS[1], lw=0.8, alpha=0.6)
    ax.plot([], [], color=SLOTS[0], lw=2, label=f"{sweep} (one line per run)")
    ax.plot([], [], color=SLOTS[1], lw=2, label="last week, reconstructed")
    ax.set_xlabel("agent step")
    ax.set_ylabel("thousand tokens sent")
    ax.set_title("Tokens sent to the model at each step (the prompt grows with the run's history)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
def write_readme(sweep: str, runs, old: list[dict]):
    now = {e["case"]: (e, r) for e, r in passing_runs(sweep, runs)}
    tot = lambda r, k: r["metrics"]["operations"][k]["time"]
    lines = [f"# Benchmark visualizations ({sweep})", "",
             "Generated by `python scripts/plot_bench.py` from `bench/index.jsonl` and each run's `record.json`. "
             "Rerun it after a sweep.", "",
             "| file | shows |", "|---|---|",
             "| `time_and_tokens_by_operation.pdf` | per run: wall time split into model generation, tool calls, "
             "Quantum ESPRESSO and the agent's scripts (minutes, then share), and tokens by the kind of call they chose |",
             "| `time_and_tokens_by_phase.pdf` | per run: time and tokens in each of the task's 7 workflow steps |",
             "| `last_week_vs_now.pdf` | the five test34 compounds, last week vs now: wall time by operation, steps, "
             "tokens |",
             "| `steps_by_sweep.pdf` | agent steps for every compound in every sweep, failures marked |",
             "| `prompt_growth.pdf` | tokens sent to the model at each step, every run |", ""]
    n = len(runs)
    llm = sum(tot(r, "llm") for _, r in runs) / sum(r["runtime"] for _, r in runs)
    qe = sum(tot(r, "qe") for _, r in runs) / sum(r["runtime"] for _, r in runs)
    lines += [f"## {sweep} in numbers", "",
              f"- {n} runs, {sum(1 for e, _ in runs if e.get('passed'))} passed; median {int(np.median([r['total_steps'] for _, r in runs]))} "
              f"steps, {np.median([r['runtime'] for _, r in runs]) / 60:.1f} min, "
              f"{np.median([r['total_tokens'] for _, r in runs]) / 1e6:.2f} M tokens per run.",
              f"- Across all runs, the model takes {llm:.0%} of wall time and Quantum ESPRESSO {qe:.0%}; "
              "tool calls and the agent's scripts take the rest.", ""]
    lines += ["## Last week vs now (same five compounds)", "",
              "| compound | steps | wall min | model min | QE min | tokens (M) |", "|---|---|---|---|---|---|"]
    for o in old:
        if o["case"] not in now:
            continue
        _, r = now[o["case"]]
        om = o["metrics"]["operations"]
        lines.append(f"| {o['case']} | {o['total_steps']} → {r['total_steps']} | {o['runtime'] / 60:.1f} → "
                     f"{r['runtime'] / 60:.1f} | {om['llm']['time'] / 60:.1f}* → {tot(r, 'llm') / 60:.1f} | "
                     f"{om['qe']['time'] / 60:.1f}* → {tot(r, 'qe') / 60:.1f} | "
                     f"{om['total']['tokens'] / 1e6:.2f}* → {r['total_tokens'] / 1e6:.2f} |")
    lines += ["", "\\* last week's runs (2026-10-07, test34 rerun on in-process transformers) predate timing and "
              "token logging. Their tokens are reconstructed by replaying each run's recorded tool calls with the "
              "exact system prompt of that commit (99.9% of the logged count when checked on sweep-8 runs); the "
              "model / tool split of their time is estimated from progress-log timestamps (on all 21 sweep-8 runs, "
              "model time within 6% median, 14% worst, of the logged value). Wall time and steps are as logged.", ""]
    failed = [e["case"] for e, _ in runs if e.get("passed") is False]
    rerun = [c for c in failed if now[c][0].get("passed")]
    if rerun:
        lines += [f"Failed in {sweep}: {', '.join(failed)}. The last-week table uses the passing follow-up rerun of "
                  f"{', '.join(rerun)}; the failed run is in the per-run charts (see `bench/RESULTS.md`).", ""]
    elif failed:
        lines += [f"Failed runs in {sweep}, included above as they ran: {', '.join(failed)} (see `bench/RESULTS.md`).", ""]
    (OUT / "README.md").write_text("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", default=None, help="label prefix of the sweep to chart (default: the latest)")
    p.add_argument("--refresh", action="store_true", help="redo last week's reconstruction")
    args = p.parse_args(argv)
    sweep = args.sweep or latest_sweep()
    runs = [(e, r) for e, r in runs_of(sweep + ":") if r.get("metrics", {}).get("operations")]
    if not runs:
        raise SystemExit(f"no runs labelled {sweep!r}")
    OUT.mkdir(parents=True, exist_ok=True)
    old = last_week(args.refresh)
    style()
    chart_operations(sweep, runs, OUT / "time_and_tokens_by_operation.pdf")
    chart_phases(sweep, runs, OUT / "time_and_tokens_by_phase.pdf")
    chart_vs_last_week(sweep, passing_runs(sweep, runs), old, OUT / "last_week_vs_now.pdf")
    chart_steps_history(OUT / "steps_by_sweep.pdf")
    chart_prompt_growth(sweep, runs, old, OUT / "prompt_growth.pdf")
    write_readme(sweep, runs, old)
    print("\n".join(str(f) for f in sorted(OUT.glob("*"))))


if __name__ == "__main__":
    main()
