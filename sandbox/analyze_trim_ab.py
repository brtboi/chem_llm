"""Compare the trim_history A/B runs written by run_trim_ab.py.

    python sandbox/analyze_trim_ab.py
"""
import json
import re
import statistics as stats
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from chem_llm.agent_core import SYSTEM_PROMPT_TEMPLATE
from chem_llm.state import AgentState

SANDBOX = Path(__file__).resolve().parent
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B", cache_dir="/pscratch/sd/b/brenthu/huggingface/hub")
T = lambda s: len(tok(s)["input_ids"])
SYS = T(SYSTEM_PROMPT_TEMPLATE)


def prompt_sizes(fs, trim):
    st, sizes = AgentState(fs["task"]), []
    for e in fs["history"]:
        sizes.append(SYS + T(st.context_summary(max_history=32, trim=trim)))
        if e["tool"] == "note":
            st.add_note(e["args"].get("text", ""))
        st.log(e["tool"], e["args"], e["result"])
    return sizes


def x_point_levels(calc):
    """Valence levels at X relative to the VBM, from the filband file."""
    dat = calc / "000.bands.dat"
    if not dat.exists():
        return None
    text = dat.read_text()
    m = re.search(r"nbnd=\s*(\d+),\s*nks=\s*(\d+)", text)
    if not m:
        return None
    nbnd, nks = map(int, m.groups())
    vals = [float(v) for v in text.split("/", 1)[1].split()]
    s = 3 + nbnd
    k = np.array([vals[i * s:i * s + 3] for i in range(nks)])
    E = np.array([vals[i * s + 3:(i + 1) * s] for i in range(nks)])
    vbm = E[:, 3].max()
    # X in fcc has |k| = 2pi/a_conv, i.e. 1/sqrt2 in QE's 2pi/alat units when alat is the
    # primitive-cell length. |k| is rotation-invariant, so this finds X whatever orientation
    # the agent's cell has; the nearest other labelled point (U) is at 0.729.
    norms = np.linalg.norm(k, axis=1)
    i = int(np.argmin(np.abs(norms - 1 / np.sqrt(2))))
    return [round(float(x - vbm), 2) for x in E[i, :6]], round(float(E[:, 4].min() - vbm), 3)


for arm in ("trim_on", "trim_off"):
    log = SANDBOX / "logs" / f"package_diamond_{arm}.jsonl"
    if not log.exists() or not log.read_text().strip():
        # run_agent creates the log at the start and appends the record at the end
        print(f"{arm}: not finished yet")
        continue
    rec = json.loads(log.read_text().strip().splitlines()[-1])
    fs = rec["final_state"]
    h = fs["history"]
    tools = [e["tool"] for e in h]
    reads = [e for e in h if e["tool"] == "read_file"]
    chars_read = sum(len(str(e["result"])) for e in reads)
    rereads, seen, dirty = 0, {}, set()
    for e in h:
        p = str(e["args"].get("path", ""))
        if e["tool"] == "read_file":
            key = (p, e["args"].get("start_line", 1), e["args"].get("num_lines"))
            if key in seen and p not in dirty:
                rereads += 1
            seen[key] = True
            dirty.discard(p)
        elif e["tool"] == "write_file":
            dirty.add(p)
    ranged = sum(1 for e in reads if "start_line" in e["args"] or "num_lines" in e["args"])
    sizes = prompt_sizes(fs, trim=(arm == "trim_on"))
    calc = SANDBOX / "package_runs" / f"diamond_{arm}" / "calculations" / "000"
    physics = x_point_levels(calc)

    print(f"\n=== {arm}: {'completed' if rec['completed'] else 'NOT completed'} in {rec['num_steps']} steps, "
          f"{rec['runtime'] / 60:.1f} min")
    print(f"  prompt tokens      median {stats.median(sizes):,.0f}   peak {max(sizes):,}")
    print(f"  read_file          {len(reads)} calls ({ranged} ranged), {chars_read:,} chars returned; "
          f"re-reads of unchanged content: {rereads}")
    print(f"  grep_file          {tools.count('grep_file')} calls; search_docs {tools.count('search_docs')}; "
          f"notes {tools.count('note')}; failed-parse steps {tools.count('parse_error')}")
    if physics:
        levels, gap = physics
        print(f"  near X, E-VBM      {levels}   (reference: -12.8 x2, -6.3 x2 | +4.7)")
        print(f"  indirect gap       {gap} eV   (reference PBE ~4.1; validated baseline 4.041)")
    else:
        print("  no bands output to check")
