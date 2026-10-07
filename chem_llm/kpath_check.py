"""Deterministic checks of a band-structure k-path, from the bands input itself.

A bands calculation's k-points are straight segments between high-symmetry
points. Two mistakes recur in agent-written paths and neither raises an
error -- they only show up as a wrong-looking band plot, or not at all:

- a segment's endpoint is not actually a high-symmetry point, usually
  because the first point after a discontinuity was dropped and the next
  segment starts one step in (a CsPbBr3 path jumped from X straight to
  (0.5, 0.5, 0.475) and never sampled R);
- the labels file does not name every endpoint (a TiO2 run labelled only the
  first occurrence of each point, so the plot mislabelled both breaks).

An endpoint passes if either (a) it is more symmetric than the path next to
it -- using the crystal's own rotations (spglib, on the very cell in the
input file, with time reversal), it is mapped to itself modulo a reciprocal
lattice vector by more operations -- or (b) it is equivalent by symmetry to
one of seekpath's named points for this structure, mapped into this cell's
frame with seekpath's rotation matrix. (b) covers points such as fcc's U and
K, which end a path where it meets the zone boundary without being more
symmetric than the line itself. Neither test depends on how the cell is
oriented. With the names from (b), a label is also checked against the point
it labels.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

__all__ = ["read_kpoints", "segment_endpoints", "check_bands_kpath"]

_TOL = 1e-5


def read_kpoints(text: str) -> tuple[str | None, np.ndarray]:
    """K_POINTS card of a pw.x input: (type, (N, 3) array of coordinates)."""
    m = re.search(r"^\s*K_POINTS\s*[{(]?\s*(\w+)?\s*[})]?", text, re.M | re.I)
    if not m:
        return None, np.zeros((0, 3))
    kind = (m.group(1) or "tpiba").lower()
    lines = text[m.end():].strip().splitlines()
    if kind in ("automatic", "gamma") or not lines or not lines[0].strip().isdigit():
        return kind, np.zeros((0, 3))
    n = int(lines[0])
    pts = []
    for line in lines[1:n + 1]:
        parts = line.split()
        try:
            pts.append([float(x.replace("d", "e").replace("D", "e")) for x in parts[:3]])
        except ValueError:
            break
    return kind, np.array(pts)


def segment_endpoints(points: np.ndarray) -> tuple[list[int], list[int]]:
    """(endpoint indices, break indices) of an explicit k-point list.

    A break is a jump much larger than the typical spacing; the points on
    both sides of it are endpoints. A turn (change of direction) is an
    endpoint; so are the first and last points. Paths often repeat the
    junction point (segment end, then the next segment's start); repeated
    points are collapsed before looking for turns, and every copy of an
    endpoint is reported.
    """
    n = len(points)
    if n < 2:
        return list(range(n)), []
    # collapse runs of identical consecutive points
    groups = [[0]]
    for i in range(1, n):
        if math.dist(points[i], points[i - 1]) < 1e-9:
            groups[-1].append(i)
        else:
            groups.append([i])
    u = np.array([points[g[0]] for g in groups])
    m = len(u)
    d = [math.dist(u[i], u[i + 1]) for i in range(m - 1)]
    positive = sorted(d)
    typical = positive[len(positive) // 2] if positive else 0.0
    ubreaks = [i + 1 for i, x in enumerate(d) if typical and x > 3 * typical]
    ends = {0, m - 1}
    for b in ubreaks:
        ends |= {b - 1, b}
    for i in range(1, m - 1):
        if i in ubreaks or i + 1 in ubreaks:
            continue
        a = (u[i] - u[i - 1]) / d[i - 1]
        w = (u[i + 1] - u[i]) / d[i]
        if np.linalg.norm(a - w) > 1e-3:
            ends.add(i)
    return sorted(j for i in ends for j in groups[i]), [groups[b][0] for b in ubreaks]


def unlabelled_endpoints(points: np.ndarray, ends: list[int], labelled: set[int], offset: int = 0) -> list[int]:
    """Endpoints with no label, counting a repeated junction point (the same
    coordinates at consecutive indices) as labelled if any copy is."""
    out = []
    for i in ends:
        copies = {i}
        for step in (-1, 1):
            j = i + step
            while 0 <= j < len(points) and math.dist(points[j], points[i]) < 1e-9:
                copies.add(j); j += step
        if not any(c + offset in labelled for c in copies):
            out.append(i)
    return out


def _labels(text: str | None) -> list[tuple[int, str]]:
    found = []
    for line in (text or "").splitlines():
        if line.strip().startswith("#"):
            continue
        toks = line.replace(",", " ").split()
        ints = [t for t in toks if re.fullmatch(r"\d+", t)]
        names = [t for t in toks if not re.fullmatch(r"[-\d.]+", t)]
        if len(ints) == 1 and names:
            found.append((int(ints[0]), names[0]))
    return found


def _equivalent(a: np.ndarray, b: np.ndarray, recip_ops: list[np.ndarray]) -> bool:
    """a and b related by a crystal rotation (with time reversal), mod G."""
    for m in recip_ops:
        for sign in (1, -1):
            diff = sign * (m @ b) - a
            if np.allclose(diff, np.round(diff), atol=1e-4):
                return True
    return False


def _named_points(atoms) -> dict[str, np.ndarray]:
    """seekpath's named points, as fractional coordinates of THIS cell's
    reciprocal basis. seekpath's rotation_matrix R takes this cell's
    cartesian frame to its standard one (v_std = R v), so k = R^T k_std."""
    import seekpath

    cell = np.array(atoms.cell[:])
    res = seekpath.get_path((cell, atoms.get_scaled_positions(), atoms.get_atomic_numbers()), symprec=1e-3)
    rot, recip = np.array(res["rotation_matrix"]), np.array(res["reciprocal_primitive_lattice"])
    return {name: (rot.T @ (np.array(f) @ recip)) @ cell.T / (2 * np.pi) for name, f in res["point_coords"].items()}


def _norm_label(name: str) -> str:
    """'$\\Gamma$', 'Γ', 'G', '\\mathrm{X}', 'x' -> 'GAMMA', 'GAMMA', 'GAMMA', 'X', 'X'."""
    n = re.sub(r"\\(?:mathrm|rm|text)\b", "", name.strip())
    n = re.sub(r"[${}\\\s]", "", n).upper()
    return "GAMMA" if n in ("G", "Γ", "GAMMA", "GM") else n


def _stabilizer_order(k_frac: np.ndarray, recip_ops: list[np.ndarray]) -> int:
    """How many operations (with time reversal) map k to itself mod G."""
    count = 0
    for m in recip_ops:
        for sign in (1, -1):
            diff = sign * (m @ k_frac) - k_frac
            if np.allclose(diff, np.round(diff), atol=_TOL):
                count += 1
    return count


def path_string(input_path: str | Path) -> str | None:
    """The k-path of a bands input as seekpath names, e.g.
    'GAMMA-X-M-GAMMA-Z-R-A-Z | X-R | M-A'. Points are named by symmetry
    from the k-point coordinates, not from any labels file; an endpoint
    that matches no named point is shown as '?'. None if not applicable."""
    input_path = Path(input_path)
    try:
        text = input_path.read_text(errors="replace")
        kind, pts = read_kpoints(text)
        if kind not in ("crystal", "tpiba") or len(pts) < 3:
            return None
        from ase.io import read
        import spglib
        atoms = read(input_path, format="espresso-in")
        cell = np.array(atoms.cell[:])
        if kind == "tpiba":
            pts = pts @ cell.T / np.linalg.norm(cell[0])
        ds = spglib.get_symmetry_dataset((cell, atoms.get_scaled_positions(), atoms.get_atomic_numbers()), symprec=1e-3)
        ops = [np.linalg.inv(r).T for r in ds.rotations] if ds is not None else [np.eye(3)]
        named = _named_points(atoms)
        import seekpath
        raw = {n: np.array(c) for n, c in seekpath.get_path(
            (cell, atoms.get_scaled_positions(), atoms.get_atomic_numbers()), symprec=1e-3)["point_coords"].items()}
    except Exception:
        return None
    ends, breaks = segment_endpoints(pts)
    out, prev = [], None
    for i in ends:
        if prev is not None and math.dist(pts[i], pts[prev]) < 1e-9:
            continue  # a repeated junction point
        # an exact match (mod G) first: equivalent points such as fcc's U and K
        # share a symmetry class but keep their own names on the path
        exact = sorted({n for pool in (raw, named) for n, k in pool.items() if _equivalent(pts[i], k, [np.eye(3)])})
        names = exact or sorted(n for n, k in named.items() if _equivalent(pts[i], k, ops))
        name = min(names, key=len) if names else "?"
        out.append((" | " if i in breaks else "-" if out else "") + name)
        prev = i
    return "".join(out)


def _check_against_kpath_json(atoms, pts: np.ndarray, kpath_json: Path) -> list[str]:
    """bands.in written from chem_llm.kpath: the cell and every k-point must
    be exactly the ones in kpath.json (which also carries the labels)."""
    import json

    kp = json.loads(kpath_json.read_text())
    issues = []
    ref_cell = np.asarray(kp["cell"])
    if not np.allclose(np.asarray(atoms.cell[:]), ref_cell, atol=1e-4):
        issues.append(
            f"k-path: the cell in this input differs from the one in {kpath_json.name}. The path's fractional "
            "k-points refer to that cell -- write kp_atoms(kp) into BOTH pw.in and bands.in.")
    ref = np.asarray(kp["kpoints"])
    if len(pts) != len(ref):
        issues.append(f"k-path: K_POINTS has {len(pts)} points but {kpath_json.name} has {len(ref)} -- write "
                      "kpts=kp_kpts(kp) unchanged.")
    elif not np.allclose(pts, ref, atol=1e-6):
        i = int(np.argmax(np.abs(pts - ref).max(axis=1)))
        issues.append(f"k-path: K_POINTS differ from {kpath_json.name} (first at index {i}: {pts[i].round(5).tolist()} "
                      f"vs {np.round(ref[i], 5).tolist()}) -- write kpts=kp_kpts(kp) unchanged.")
    return issues


def check_bands_kpath(input_path: str | Path, labels_path: str | Path | None = None) -> list[str]:
    """Issues with the k-path of a pw.x bands input (calculation='bands').
    Returns [] for inputs this does not apply to (SCF meshes, *_b lists)."""
    input_path = Path(input_path)
    text = input_path.read_text(errors="replace")
    if not re.search(r"calculation\s*=\s*['\"]bands['\"]", text, re.I):
        return []
    kind, pts = read_kpoints(text)
    if kind not in ("crystal", "tpiba") or len(pts) < 3:
        return []

    try:
        from ase.io import read
        import spglib
        atoms = read(input_path, format="espresso-in")
    except Exception as e:  # unreadable structure: report, don't guess
        return [f"k-path: could not read the structure from {input_path.name} to check the path ({e})"]

    kpath_json = input_path.parent / "kpath.json"
    if kpath_json.exists():
        return _check_against_kpath_json(atoms, pts, kpath_json)

    cell = np.array(atoms.cell[:])
    if kind == "tpiba":
        # cartesian in 2pi/alat; QE's alat for ibrav=0 is |a1| (cell in angstrom here)
        alat = np.linalg.norm(cell[0])
        pts = pts @ cell.T / alat
    ds = spglib.get_symmetry_dataset((cell, atoms.get_scaled_positions(), atoms.get_atomic_numbers()), symprec=1e-3)
    # direct-lattice rotations act on reciprocal fractional coordinates as (R^-1)^T
    recip_ops = [np.linalg.inv(r).T for r in ds.rotations] if ds is not None else [np.eye(3)]

    try:
        named = _named_points(atoms)
    except Exception:
        named = {}
    ends, breaks = segment_endpoints(pts)
    issues = []

    not_special, names_at = [], {}
    for i in ends:
        names_at[i] = {n for n, k in named.items() if _equivalent(pts[i], k, recip_ops)}
        if names_at[i]:
            continue  # one of seekpath's named points (or equivalent to one)
        order = _stabilizer_order(pts[i], recip_ops)
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < len(pts) and j not in ends
                      and not (j == i + 1 and i + 1 in breaks) and not (j == i - 1 and i in breaks)]
        if neighbours and order <= max(_stabilizer_order(pts[j], recip_ops) for j in neighbours):
            not_special.append(i)
    if not_special:
        where = ", ".join(f"{i} at ({', '.join(f'{x:.4g}' for x in pts[i])})" for i in not_special[:6])
        issues.append(
            f"k-path: segment endpoint(s) {where} are not high-symmetry points -- no more symmetric than the "
            "path next to them. A segment's start or end point is missing: the usual cause is that the first "
            "point of a segment after a discontinuity was not added, so the segment starts one step in. Every "
            "segment must contribute its own start point (INVARIANT 3)."
        )

    if labels_path is not None and Path(labels_path).exists():
        entries = _labels(Path(labels_path).read_text(errors="replace"))
        labelled = {i for i, _ in entries}
        wrong = []
        for i, name in entries:
            expected = {_norm_label(n) for n in names_at.get(i, set())}
            given = {_norm_label(part) for part in re.split(r"[|/,]", name) if part}
            if expected and not (given & expected):
                wrong.append(f"index {i} is labelled {name} but the k-point there is {' / '.join(sorted(expected))}")
        if wrong:
            issues.append(f"k-path: {Path(labels_path).name} mislabels points: " + "; ".join(wrong[:6]) + ".")
        missing = unlabelled_endpoints(pts, ends, labelled)
        shifted = unlabelled_endpoints(pts, ends, labelled, offset=1)  # a 1-based labels file
        if missing and shifted:
            issues.append(
                f"k-path: {Path(labels_path).name} does not label segment endpoint(s) at k-point index "
                f"{', '.join(map(str, missing))} (the path has {len(ends)} endpoints: "
                f"{', '.join(map(str, ends))}). Write one entry per endpoint, including repeated points such as "
                "a second GAMMA and both sides of every discontinuity."
            )
    return issues
