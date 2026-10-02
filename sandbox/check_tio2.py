"""Check a rutile TiO2 band structure against the reference runs.

    python sandbox/check_tio2.py sandbox/test29

Rutile's gap is direct at Gamma, so the gap alone cannot catch a wrong k-path.
This compares band energies at every high-symmetry point of the tetragonal
zone (Gamma, X, M, Z, R, A), found by their coordinates in the bands output
rather than by the run's own label file, against each reference run.

Energies are relative to the VBM; compared bands are the top 4 valence and
bottom 4 conduction bands. The references use different pseudopotentials, so
the reference-vs-reference spread (printed first) is the natural tolerance.
"""
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
REFERENCES = {
    "test_claude": REPO / "sandbox/test_claude/calculations/000",
    "test_claude_api": REPO / "sandbox/test_claude_api/calculations/000",
    "test27": REPO / "sandbox/test27/calculations/000",
}
POINTS = ("GAMMA", "X", "M", "Z", "R", "A")
N_VAL, N_COND = 4, 4


def load(calc: Path) -> dict:
    """Band energies (nks, nbnd), k-points (cartesian, 2pi/alat) and cell info."""
    pw = (calc / "pw.out").read_text(errors="replace")
    electrons = float(re.search(r"number of electrons\s*=\s*([\d.]+)", pw).group(1))
    num = r"\s*([-\d.]+)"
    axes = np.array([[float(x) for x in re.search(rf"a\({i}\) = \({num}{num}{num}", pw).groups()] for i in (1, 2, 3)])
    text = (calc / "000.bands.dat").read_text()
    nbnd, nks = map(int, re.search(r"nbnd=\s*(\d+),\s*nks=\s*(\d+)", text).groups())
    vals = [float(v) for v in text.split("/", 1)[1].split()]
    s = 3 + nbnd
    k = np.array([vals[i * s:i * s + 3] for i in range(nks)])
    E = np.array([vals[i * s + 3:(i + 1) * s] for i in range(nks)])
    n_occ = int(round(electrons / 2))
    return {"k": k, "E": E, "n_occ": n_occ, "axes": axes, "nbnd": nbnd, "nks": nks}


def tetragonal_c_axis(axes: np.ndarray) -> int:
    """Index of the unique-length axis of a tetragonal cell (lengths a, a, c
    in any order and orientation); raises if the cell is not tetragonal."""
    lengths = np.linalg.norm(axes, axis=1)
    angles_ok = all(abs(axes[i] @ axes[j]) < 1e-3 * lengths[i] * lengths[j] for i, j in ((0, 1), (0, 2), (1, 2)))
    for c in range(3):
        a, b = [i for i in range(3) if i != c]
        if angles_ok and abs(lengths[a] - lengths[b]) < 1e-4 and abs(lengths[c] - lengths[a]) > 1e-3:
            return c
    raise ValueError(f"cell is not tetragonal (axis lengths {np.round(lengths, 4)}); cannot locate rutile's points")


def special_points(run: dict) -> dict:
    """Energies (relative to VBM) of the bands around the gap at each point,
    matched by a key that is invariant under the tetragonal point group."""
    # Fractional coordinates in the run's own reciprocal basis (k is cartesian in
    # 2pi/alat, axes are in alat, so frac = k . axes^T), reordered so the c axis
    # is last and folded into [0, 0.5]: independent of how the cell is oriented.
    c = tetragonal_c_axis(run["axes"])
    order = [i for i in range(3) if i != c] + [c]
    frac = (run["k"] @ run["axes"].T)[:, order]
    frac = np.abs(frac - np.round(frac))
    targets = {"GAMMA": (0, 0, 0), "X": (0, 0.5, 0), "M": (0.5, 0.5, 0),
               "Z": (0, 0, 0.5), "R": (0, 0.5, 0.5), "A": (0.5, 0.5, 0.5)}
    key = lambda v: (*sorted((v[0], v[1])), v[2])
    vbm = run["E"][:, run["n_occ"] - 1].max()
    lo, hi = run["n_occ"] - N_VAL, run["n_occ"] + N_COND
    found = {}
    for name, t in targets.items():
        hits = [i for i, kv in enumerate(frac) if np.allclose(key(kv), key(t), atol=2e-3)]
        if hits:
            found[name] = run["E"][hits[0], lo:hi] - vbm
    return found


def gap(run: dict) -> tuple[float, str]:
    v, c = run["E"][:, run["n_occ"] - 1], run["E"][:, run["n_occ"]]
    direct = v.argmax() == c.argmin()
    return float(c.min() - v.max()), "direct" if direct else "indirect"


def compare(a: dict, b: dict) -> tuple[float, list[str]]:
    pa, pb = special_points(a), special_points(b)
    common = [p for p in POINTS if p in pa and p in pb]
    worst = max((float(np.abs(pa[p] - pb[p]).max()) for p in common), default=float("nan"))
    return worst, [p for p in POINTS if p not in common]


def main(run_dir: Path):
    refs = {name: load(path) for name, path in REFERENCES.items()}
    print("reference spread (max |dE| over points and the 8 bands around the gap):")
    for (n1, r1), (n2, r2) in combinations(refs.items(), 2):
        worst, missing = compare(r1, r2)
        print(f"  {n1:16} vs {n2:16} {worst:6.3f} eV" + (f"   missing: {missing}" if missing else ""))
    spread = max(compare(r1, r2)[0] for (_, r1), (_, r2) in combinations(refs.items(), 2))

    calc = Path(run_dir) / "calculations" / "000"
    if not (calc / "000.bands.dat").exists():
        print(f"\n{run_dir}: no 000.bands.dat -- bands did not finish")
        return 1
    run = load(calc)
    g, kind = gap(run)
    try:
        pts = special_points(run)
    except ValueError as e:
        print(f"\n{run_dir}: {e}\nverdict: DIFFERS from the references -- inspect")
        return 1
    lengths = np.linalg.norm(run["axes"], axis=1)
    print(f"\n{run_dir}: {run['n_occ'] * 2} electrons, {run['nbnd']} bands, {run['nks']} k-points, "
          f"axis lengths {np.round(lengths, 4)} (alat)")
    print(f"  gap {g:.3f} eV ({kind})   references: " + ", ".join(f"{n} {gap(r)[0]:.3f}" for n, r in refs.items()))
    print(f"  high-symmetry points found: {[p for p in POINTS if p in pts]}")
    for p in POINTS:
        if p in pts:
            print(f"    {p:5} " + " ".join(f"{x:7.2f}" for x in pts[p]))
    ok = True
    for name, ref in refs.items():
        worst, missing = compare(run, ref)
        flag = "OK" if worst <= max(2 * spread, 0.15) and not missing else "CHECK"
        ok &= flag == "OK"
        print(f"  vs {name:16} max |dE| {worst:.3f} eV   {flag}" + (f"   missing: {missing}" if missing else ""))
    print(f"\nverdict: {'matches the references' if ok else 'DIFFERS from the references -- inspect'} "
          f"(tolerance {max(2 * spread, 0.15):.3f} eV = 2x reference spread, min 0.15)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "sandbox/test29"))
