"""Materials Project's own band structure: a reference to compare against,
the k-path it was computed on, and how MP set up the calculation.

    ref = fetch_reference("mp-2657", api_key=..., cache_dir=...)   # compact, JSON-able
    kp  = mp_kpath(ref)          # same dict format as kpath.standard_kpath:
                                 # kp_atoms / kp_kpts / kp_save / plot_axis all work
    score = compare_bands(kp, energies, ref, n_occupied=..., fermi_eV=...)

MP's line-mode band structures are VASP GGA (PBE, or PBE+U for some
transition-metal oxides and fluorides) NSCF runs on MP's primitive cell, with
no spin-orbit coupling. Their k-points are fractions of THAT cell's reciprocal
lattice, so a calculation meant to be compared point by point must run on the
cell in the reference (mp_kpath's cell), not a re-standardised one.

The VASP POTCARs MP used are named in `calculation.potcars` (e.g. "PAW_PBE
Ti_pv 07Sep2000"). Quantum ESPRESSO cannot read them; the closest match is a
PBE PAW pseudopotential with the same semicore states in valence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["fetch_reference", "mp_kpath", "compare_bands", "potcar_valence_note", "calculation_summary"]

REFERENCE_VERSION = 1

# VASP POTCAR suffixes -> which states are treated as valence
_POTCAR_SUFFIX = {
    "pv": "semicore p states in valence",
    "sv": "semicore s and p states in valence",
    "d": "semicore d states in valence",
    "h": "hard (small core radius)",
    "s": "soft (large core radius)",
    "GW": "GW-optimised",
    "3": "f electrons frozen in the core (trivalent)",
    "2": "f electrons frozen in the core (divalent)",
}


def potcar_valence_note(titel: str) -> dict:
    """{'potcar', 'element', 'variant', 'meaning'} from a POTCAR TITEL such as
    'PAW_PBE Ti_pv 07Sep2000'."""
    parts = titel.split()
    name = parts[1] if len(parts) > 1 else titel
    element, _, variant = name.partition("_")
    meanings = [_POTCAR_SUFFIX.get(v, v) for v in variant.split("_") if v] if variant else []
    return {"potcar": titel, "element": element, "variant": name,
            "meaning": "; ".join(meanings) or "standard valence (no semicore states)"}


def _label(raw: str) -> str:
    """pymatgen's '\\Gamma' / '\\Sigma_1' -> 'GAMMA' / 'SIGMA_1' (seekpath's spelling)."""
    if raw.startswith("\\"):
        head, _, tail = raw[1:].partition("_")
        return head.upper() + ("_" + tail if tail else "")
    return raw


def _cache_path(cache_dir: Path, material_id: str, path_type: str) -> Path:
    return Path(cache_dir) / f"{material_id}_{path_type}.json"


def fetch_reference(material_id: str, path_type: str = "setyawan_curtarolo", *, api_key: str | None = None,
                    cache_dir: str | Path | None = None, refresh: bool = False) -> dict:
    """MP's line-mode band structure for `material_id`, with the cell and the
    calculation settings, as a compact JSON-able dict (cached in cache_dir).
    Raises LookupError if MP has no band structure for it."""
    if cache_dir is not None and not refresh:
        cached = _cache_path(cache_dir, material_id, path_type)
        if cached.exists():
            ref = json.loads(cached.read_text())
            if ref.get("version") == REFERENCE_VERSION:
                return ref

    from emmet.core.band_theory import BSPathType
    from mp_api.client import MPRester
    from pymatgen.electronic_structure.core import Spin

    with MPRester(api_key, mute_progress_bars=True) as mpr:
        try:
            bs = mpr.get_bandstructure_by_material_id(material_id, path_type=BSPathType(path_type))
        except Exception as e:
            raise LookupError(f"Materials Project has no {path_type} band structure for {material_id}: {e}") from e
        if bs is None:
            raise LookupError(f"Materials Project has no {path_type} band structure for {material_id}")
        task_id = None
        try:
            es = mpr.materials.electronic_structure.search(material_ids=[material_id])[0]
            task_id = str(getattr(es.bandstructure, path_type).task_id)
        except Exception:
            pass
        calc = {}
        if task_id:
            try:
                t = mpr.materials.tasks.search(task_ids=[task_id], fields=["input", "calc_type", "run_type"])[0]
                incar = dict(t.input.incar or {})
                calc = {
                    "task_id": task_id, "calc_type": str(t.calc_type), "run_type": str(t.run_type),
                    "potcars": [p.titel for p in (t.input.potcar_spec or [])],
                    "hubbard_U_eV": dict(t.input.hubbards or {}),
                    "incar": {k: incar[k] for k in ("ENCUT", "ISPIN", "ISMEAR", "SIGMA", "LDAU", "LDAUU", "LDAUL",
                                                    "LSORBIT", "GGA", "METAGGA", "MAGMOM", "NBANDS") if k in incar},
                }
            except Exception as e:
                calc = {"task_id": task_id, "error": f"{type(e).__name__}: {e}"}

    s = bs.structure
    spins = {"up": Spin.up, "down": Spin.down}
    energies = {name: np.round(bs.bands[spin].T, 4).tolist() for name, spin in spins.items() if spin in bs.bands}
    metal = bool(bs.is_metal())
    vbm = None if metal else bs.get_vbm()
    cbm = None if metal else bs.get_cbm()
    vbm_band = None
    if vbm is not None:
        vbm_band = max(i for idx in vbm["band_index"].values() for i in idx)
    ref = {
        "version": REFERENCE_VERSION,
        "material_id": material_id,
        "path_type": path_type,
        "formula": s.composition.reduced_formula,
        "cell": np.asarray(s.lattice.matrix).tolist(),
        "symbols": [site.specie.symbol for site in s],
        "scaled_positions": np.asarray(s.frac_coords).tolist(),
        "kpoints": [np.asarray(k.frac_coords).round(8).tolist() for k in bs.kpoints],
        "labels": [_label(k.label) if k.label else "" for k in bs.kpoints],
        "branches": [{"start": b["start_index"], "end": b["end_index"],
                      "name": "-".join(_label(x) for x in b["name"].split("-"))} for b in bs.branches],
        "energies": energies,               # {"up": (nk, nb) eV, "down": ...}, absolute (VASP)
        "efermi": float(bs.efermi),
        "is_metal": metal,
        "is_spin_polarized": bool(bs.is_spin_polarized),
        "band_gap_eV": 0.0 if metal else float(bs.get_band_gap()["energy"]),
        "vbm_eV": None if vbm is None else float(vbm["energy"]),
        "cbm_eV": None if cbm is None else float(cbm["energy"]),
        "vbm_band": vbm_band,               # 0-based index of the highest occupied band (spin up)
        "calculation": calc,
    }
    if cache_dir is not None:
        path = _cache_path(cache_dir, material_id, path_type)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ref))
    return ref


def calculation_summary(ref: dict) -> dict:
    """How MP ran this band structure, phrased for the agent's pseudopotential
    and Hubbard choices."""
    calc = ref.get("calculation") or {}
    out = {
        "material_id": ref["material_id"],
        "code": "VASP (PAW)",
        "functional": "PBE+U" if calc.get("hubbard_U_eV") else "PBE",
        "spin_orbit": bool((calc.get("incar") or {}).get("LSORBIT")),
        "spin_polarized": ref.get("is_spin_polarized"),
        "potcars": [potcar_valence_note(t) for t in calc.get("potcars", [])],
        "hubbard_U_eV": calc.get("hubbard_U_eV") or {},
        "plane_wave_cutoff_eV": (calc.get("incar") or {}).get("ENCUT"),
        "band_gap_eV": ref.get("band_gap_eV"),
    }
    if ref.get("vbm_band") is not None and not ref.get("is_spin_polarized"):
        # fully occupied bands x 2: the valence electrons MP's POTCARs give this cell
        out["valence_electrons_per_cell"] = 2 * (ref["vbm_band"] + 1)
    return out


def mp_kpath(ref: dict) -> dict:
    """The reference's path as a kpath dict (the format standard_kpath
    returns): the cell, k-points, cumulative x (no distance across a break),
    labels, ticks, continuous segments and a readable path."""
    from .kpath import _ticks

    cell = np.asarray(ref["cell"], dtype=float)
    recip = 2 * np.pi * np.linalg.inv(cell).T
    k = np.asarray(ref["kpoints"], dtype=float)
    cart = k @ recip
    starts = {b["start"] for b in ref["branches"]}
    x = [0.0]
    for i in range(1, len(k)):
        # a new branch starts at i: no distance from the previous branch's end
        # (the same point repeated, or a jump across a discontinuity)
        x.append(x[-1] + (0.0 if i in starts else float(np.linalg.norm(cart[i] - cart[i - 1]))))
    labels = list(ref["labels"])
    runs: list[list[int]] = []
    for b in ref["branches"]:
        a, e = b["start"], b["end"] + 1
        continuous = runs and np.allclose(k[runs[-1][1] - 1], k[a], atol=1e-6)
        if continuous:
            runs[-1][1] = e
        else:
            runs.append([a, e])
    names: list[list[str]] = []
    for b in ref["branches"]:
        first, last = b["name"].split("-")[0], b["name"].split("-")[-1]
        if names and np.allclose(k[b["start"]], k[b["start"] - 1], atol=1e-6):
            names[-1].append(last)
        else:
            names.append([first, last])
    return {
        "cell": cell.tolist(),
        "symbols": list(ref["symbols"]),
        "scaled_positions": ref["scaled_positions"],
        "kpoints": k.tolist(),
        "x": x,
        "labels": labels,
        "ticks": _ticks(labels, x),
        "segments": [tuple(r) for r in runs],
        "path": " | ".join("-".join(n) for n in names),
        "spacegroup": None,
        "source": f"materials_project:{ref['material_id']}:{ref['path_type']}",
        "density_per_angstrom": None,
    }


# ----------------------------------------------------------------------
# scoring
def _run_at_reference(kp: dict, energies: np.ndarray, ref: dict, tol: float = 2e-3):
    """The run's band energies at the reference's k-points: (reference
    indices, energies (n, nbnd), how). Same path -> every point as is.
    Otherwise each reference point that lies on one of the run's path
    segments (cartesian, 1/A) is linearly interpolated between the two run
    points around it -- same path at another density, or just the shared
    stretches of a different path."""
    ka, kr = np.asarray(kp["kpoints"], float), np.asarray(ref["kpoints"], float)
    ca, cr = np.asarray(kp["cell"], float), np.asarray(ref["cell"], float)
    if ka.shape == kr.shape and np.allclose(ka, kr, atol=1e-5) and np.allclose(ca, cr, atol=1e-3):
        return list(range(len(kr))), energies, "same path"
    recip_r = 2 * np.pi * np.linalg.inv(cr).T
    # the same cell up to a small strain (a relaxed or re-fetched structure):
    # the same fractional k is the same symmetry point, so place both in the
    # reference's reciprocal lattice; otherwise compare true cartesian k
    strained = np.linalg.norm(ca - cr) / np.linalg.norm(cr) < 0.05
    xa = ka @ (recip_r if strained else 2 * np.pi * np.linalg.inv(ca).T)
    xr = kr @ recip_r
    pairs = [(i, i + 1) for a, b in kp["segments"] for i in range(a, b - 1)]
    idx, rows = [], []
    for n, v in enumerate(xr):
        d = np.linalg.norm(xa - v, axis=1)
        m = int(np.argmin(d))
        if d[m] < tol:  # a run point sits right there
            idx.append(n)
            rows.append(energies[m])
            continue
        for i, k in pairs:
            step = xa[k] - xa[i]
            seg2 = float(step @ step)
            if seg2 == 0:
                continue
            t = float((v - xa[i]) @ step) / seg2
            # on the segment (projection inside it) and within tol of the line
            if -1e-6 <= t <= 1 + 1e-6 and np.linalg.norm(xa[i] + t * step - v) < tol:
                t = min(max(t, 0.0), 1.0)
                idx.append(n)
                rows.append((1 - t) * energies[i] + t * energies[k])
                break
    how = "interpolated along the run's path" if len(idx) > 0.9 * len(kr) else "shared stretches of the path only"
    if strained and not np.allclose(ca, cr, atol=1e-3):
        how += f" (cells differ by {100 * np.linalg.norm(ca - cr) / np.linalg.norm(cr):.1f}%: matched by fractional k)"
    return idx, np.asarray(rows).reshape(len(rows), energies.shape[1]), how


def compare_bands(kp: dict, energies, ref: dict, *, n_occupied: int | None = None, fermi_eV: float | None = None,
                  below_eV: float = 6.0, above_eV: float = 3.0, spin: str = "up") -> dict:
    """Band energies of a run (on kpath `kp`, shape (nk, nbnd), eV) against
    MP's at the same k-points.

    Both sides are aligned to their own VBM (insulators; n_occupied gives the
    run's) or Fermi energy (metals; fermi_eV). MP band j is paired with run
    band j + offset: for an insulator the offset lines the two VBMs up (the
    codes may put different semicore states in valence); for a metal it is
    the offset with the lowest RMS. Bands compared: MP bands reaching into
    [zero - below_eV, top + above_eV], top = CBM (insulator) or E_F (metal).
    Returns rms/mae/max (eV), the gap of each, and what was compared.
    """
    E = np.asarray(energies, float)
    R = np.asarray(ref["energies"].get(spin) or ref["energies"]["up"], float)
    ir, Ea, how = _run_at_reference(kp, E, ref)
    out: dict = {"material_id": ref["material_id"], "path_type": ref["path_type"], "k_matching": how,
                 "n_kpoints": len(ir), "n_kpoints_reference": len(ref["kpoints"]),
                 "mp_gap_eV": ref.get("band_gap_eV"), "rms_eV": None}
    if len(ir) < 2:
        out["detail"] = (f"only {len(ir)} k-point(s) of Materials Project's {ref['path_type']} path lie on the "
                         "run's path: no band RMS")
        return out
    Rr = R[ir]
    insulator = not ref.get("is_metal") and n_occupied is not None and 0 < n_occupied < E.shape[1]
    if insulator:
        zero_a = float(E[:, n_occupied - 1].max())
        zero_r = float(ref["vbm_eV"])
        top_r = float(ref["cbm_eV"]) - zero_r
        out["run_gap_eV"] = round(float(E[:, n_occupied].min() - zero_a), 4)
        offsets = [n_occupied - 1 - ref["vbm_band"]]
    else:
        if fermi_eV is None:
            out["detail"] = "metal (per MP) but the run has no Fermi energy to align to: no band RMS"
            return out
        zero_a, zero_r, top_r = float(fermi_eV), float(ref["efermi"]), 0.0
        offsets = list(range(-12, 13))
    Ra, Aa = Rr - zero_r, Ea - zero_a
    bands = [j for j in range(R.shape[1]) if Ra[:, j].max() >= -below_eV and Ra[:, j].min() <= top_r + above_eV]
    best = None
    for o in offsets:
        use = [j for j in bands if 0 <= j + o < E.shape[1]]
        if not use:
            continue
        d = Aa[:, [j + o for j in use]] - Ra[:, use]
        rms = float(np.sqrt(np.mean(d ** 2)))
        if best is None or rms < best[0]:
            best = (rms, o, use, d)
    if best is None:
        out["detail"] = "no bands in common within the energy window"
        return out
    rms, o, use, d = best
    out.update({
        "rms_eV": round(rms, 4),
        "mae_eV": round(float(np.mean(np.abs(d))), 4),
        "max_abs_eV": round(float(np.max(np.abs(d))), 4),
        "band_offset": o,
        "n_bands": len(use),
        "aligned_to": "vbm" if insulator else "fermi",
        "window_eV": [-below_eV, round(top_r + above_eV, 3)],
        "per_band_rms_eV": {str(j - (ref.get("vbm_band") or 0)) if insulator else str(j):
                            round(float(np.sqrt(np.mean(d[:, n] ** 2))), 4) for n, j in enumerate(use)},
    })
    if insulator:
        out["gap_error_eV"] = round(out["run_gap_eV"] - ref["band_gap_eV"], 4)
    out["detail"] = (f"RMS {rms:.3f} eV over {len(use)} bands x {len(ir)}/{len(ref['kpoints'])} MP k-points "
                     f"({how}; aligned to the "
                     f"{'VBM' if insulator else 'Fermi energy'}, band offset {o:+d})")
    return out


def sidecar(path: str | Path) -> dict:
    """The materials_project.json generate_cif writes beside a structure."""
    p = Path(path)
    if p.is_dir():
        p = p / "materials_project.json"
    elif p.name != "materials_project.json":  # a structure file: look beside it
        p = p.parent / "materials_project.json"
    return json.loads(p.read_text()) if p.exists() else {}


def score_run(work_dir: str | Path, *, path_type: str = "setyawan_curtarolo", api_key: str | None = None,
              cache_dir: str | Path | None = None, job: str = "000") -> dict | None:
    """compare_bands for a finished run's calculations/<job>, against the
    Materials Project entry its structures/materials_project.json names.
    A run on MP's path is scored against that path; any other run against
    MP's `path_type` band structure, its bands interpolated onto MP's
    k-points. None if the run has no MP entry or no bands to score."""
    from .kpath import load_kpath, plot_axis
    from .qe_output import read_pw_out

    work_dir = Path(work_dir)
    calc = work_dir / "calculations" / job
    material_id = sidecar(work_dir / "structures").get("material_id")
    if not material_id or not (calc / "kpath.json").exists():
        return None
    kp = load_kpath(calc / "kpath.json")
    source = kp.get("source") or ""
    if source.startswith("materials_project:"):
        path_type = source.split(":")[2]
    try:
        ax = plot_axis(calc / "kpath.json", calc / f"{job}.bands.dat")
        pw = read_pw_out(calc / "pw.out")
    except (OSError, ValueError, KeyError) as e:
        return {"material_id": material_id, "rms_eV": None, "detail": f"no bands to score: {e}"}
    ref = fetch_reference(material_id, path_type, api_key=api_key, cache_dir=cache_dir)
    # MP decides insulator vs metal: an insulator run with smearing is still
    # aligned at its VBM, found by counting its occupied bands
    out = compare_bands(kp, ax["energies"], ref, n_occupied=pw.get("n_occupied"), fermi_eV=pw.get("fermi_energy_eV"))
    out["run_path"] = "materials_project" if source.startswith("materials_project:") else "seekpath"
    out["run_spin_orbit"] = bool(pw.get("spin_orbit"))
    if pw.get("spin_orbit"):
        out["detail"] += " -- the run has spin-orbit, MP's bands do not"
    km = calc / "kmesh_convergence.json"
    if km.exists():
        k = json.loads(km.read_text())
        out["kmesh"] = {"mesh": k.get("mesh"), "converged": k.get("converged"), "n_scf": len(k.get("meshes", []))}
    return out
