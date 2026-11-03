"""Reading a finished chem_llm QE run and writing a DeePseudopot input bundle.

The bookkeeping between the two formats is deterministic, so it lives here
rather than in agent-written code (the same split as chem_llm.kpath /
qe_output for the DFT agent). The physics decisions -- which DFT bands the
empirical pseudopotential should reproduce, which k-points, weights, basis
cutoff, training length -- stay with the agent.

    from chem_llm.dpp_io import read_qe_run, pseudo_orbitals, write_bundle, read_fit

    run = read_qe_run("qe_run")            # structure, bands, k-path, electron count, SOC
    pseudo_orbitals("qe_run/Ti.upf")       # [{"label": "3S", "l": 0, "occupation": 2.0}, ...]
    write_bundle("nn_inputs", run, first_band=8, n_bands=26, kpoint_indices=[...],
                 band_weights=[...], input_params={...}, nn_config={...}, init_params={...})
    read_fit("dpp_results", "nn_inputs")   # NN vs DFT: RMS error, gap, per-band errors

Units: DeePseudopot wants the cell in Bohr, energies in eV, maxKE in
Hartree, k-points in fractional coordinates of the cell's reciprocal basis.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


def read_qe_run(calc_dir: str | Path) -> dict:
    """A chem_llm QE run directory (pw.in, pw.out, kpath.json, the filband
    *.bands.dat) as one dict:

      symbols, cell_angstrom (3x3), scaled_positions   the cell pw.x ran on
      n_electrons, n_occupied, spin_orbit, noncollinear  from pw.out
      energies       (n_kpoints, n_bands) eV, from bands.x's filband
      kpoints        (n_kpoints, 3) fractional, x (path length), labels ('' between)
      high_symmetry  [(index, label)] of the labelled points, in path order
      vbm, cbm, gap  over every k-point (eV), from band_edges
      pseudopotentials  {element: UPF path} from pw.in's ATOMIC_SPECIES
    """
    from ase.io import read

    from .kpath import _read_filband, load_kpath
    from .qe_output import band_edges, read_pw_out

    d = Path(calc_dir)
    atoms = read(d / "pw.in", format="espresso-in")
    pw = read_pw_out(d / "pw.out")
    kp = load_kpath(d / "kpath.json")
    filband = sorted(p for p in d.glob("*.bands.dat") if not p.name.endswith(".gnu"))
    if not filband:
        raise FileNotFoundError(f"no *.bands.dat (bands.x filband) in {d}")
    _, energies = _read_filband(filband[0])
    if len(energies) != len(kp["kpoints"]):
        raise ValueError(f"{filband[0]} has {len(energies)} k-points, kpath.json {len(kp['kpoints'])}")
    if not np.allclose(atoms.cell[:], np.asarray(kp["cell"]), atol=1e-4):
        raise ValueError("pw.in's cell differs from kpath.json's: the k-points would refer to another cell")
    edges = band_edges(energies, pw["n_occupied"])
    species = re.search(r"ATOMIC_SPECIES\s*\n(.*?)\n\s*\n", (d / "pw.in").read_text() + "\n\n", re.S | re.I)
    pseudos = {}
    if species:
        for line in species.group(1).splitlines():
            parts = line.split()
            if len(parts) >= 3 and (d / parts[2]).exists():
                pseudos[parts[0]] = str((d / parts[2]).resolve())
    return {
        "calc_dir": str(d.resolve()),
        "symbols": atoms.get_chemical_symbols(),
        "cell_angstrom": atoms.cell[:].tolist(),
        "scaled_positions": atoms.get_scaled_positions().tolist(),
        "n_electrons": pw["n_electrons"],
        "n_occupied": pw["n_occupied"],
        "spin_orbit": pw["spin_orbit"],
        "noncollinear": pw["noncollinear"],
        "energies": energies,
        "kpoints": np.asarray(kp["kpoints"]),
        "x": np.asarray(kp["x"]),
        "labels": list(kp["labels"]),
        "high_symmetry": [(i, lab) for i, lab in enumerate(kp["labels"]) if lab],
        "vbm": edges.get("vbm"), "cbm": edges.get("cbm"), "gap": edges.get("gap"),
        "pseudopotentials": pseudos,
    }


def pseudo_orbitals(upf_path: str | Path) -> list[dict]:
    """The valence orbitals a UPF pseudopotential treats explicitly, from its
    PP_PSWFC/PP_CHI entries: [{"label": "3S", "l": 0, "occupation": 2.0}].
    Their occupations sum to the file's z_valence; deep ones (semicore,
    e.g. Ti 3s/3p) are what DFT bands an empirical pseudopotential omits."""
    text = Path(upf_path).read_text(errors="replace")
    out = []
    for m in re.finditer(r"<PP_CHI\.\d+(.*?)>", text, re.S):  # UPF v2
        a = dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', m.group(1)))
        out.append({"label": a.get("label", "?").upper(), "l": int(a.get("l", -1)),
                    "occupation": float(a.get("occupation", "nan"))})
    if not out:  # UPF v1: "<PP_PSWFC>" then "<label> <l> <occupation> Wavefunction" lines
        block = re.search(r"<PP_PSWFC>(.*?)</PP_PSWFC>", text, re.S)
        for line in (block.group(1).splitlines() if block else []):
            p = line.split()
            if len(p) >= 4 and p[-1].lower() == "wavefunction":
                out.append({"label": p[0].upper(), "l": int(p[1]), "occupation": float(p[2])})
    return out


def _check(cond, msg):
    if not cond:
        raise ValueError(msg)


def write_bundle(out_dir: str | Path, run: dict, *, first_band: int, n_bands: int, kpoint_indices,
                 band_weights, input_params: dict, nn_config: dict, init_params: dict,
                 energy_shift: float = 0.0) -> dict:
    """Write a complete DeePseudopot input bundle for one system (index 0).

    first_band      the DFT band (0-based) that becomes the empirical model's
                    lowest band: every DFT band below it is dropped
    n_bands         how many DFT bands, from first_band up, the model fits
    kpoint_indices  which of run's k-points to fit (indices into run["kpoints"])
    band_weights    n_bands per-band weights for the loss
    input_params    input_0.par keys (maxKE in Hartree, idxVB/idxCB relative to
                    first_band, systemName, BS_plot_*); nBands is filled in
    nn_config       NN_config.par keys (num_cores must be 0; SObool must match run)
    init_params     {element: 9 Zunger-form floats} for init_<element>Params.par
    energy_shift    added to every energy (eV), e.g. -run["vbm"] to put the VBM at 0

    Validates the bundle against itself before writing and returns a summary.
    """
    out = Path(out_dir)
    E, kpts = np.asarray(run["energies"]), np.asarray(run["kpoints"])
    kidx = [int(i) for i in kpoint_indices]
    _check(0 <= first_band and first_band + n_bands <= E.shape[1],
           f"bands {first_band}..{first_band + n_bands - 1} are outside the {E.shape[1]} DFT bands computed")
    _check(kidx and all(0 <= i < len(kpts) for i in kidx), f"kpoint_indices must index the {len(kpts)} k-points")
    _check(len(band_weights) == n_bands, f"band_weights has {len(band_weights)} entries, n_bands is {n_bands}")
    species = list(dict.fromkeys(run["symbols"]))
    _check(set(init_params) == set(species), f"init_params must have exactly {species}, got {sorted(init_params)}")
    for el, p in init_params.items():
        _check(len(p) == 9, f"init_params[{el!r}] needs the 9 Zunger-form parameters, got {len(p)}")
    cfg = {k: v for k, v in nn_config.items()}
    _check(int(cfg.get("num_cores", 0)) == 0, "num_cores must be 0 (>=1 crashes and runs out of memory)")
    so = bool(int(cfg.get("SObool", 0)))
    _check(so == bool(run["spin_orbit"]),
           f"SObool = {int(so)} but the DFT run {'has' if run['spin_orbit'] else 'has no'} spin-orbit coupling")
    params = dict(input_params)
    _check("maxKE" in params, "input_params needs maxKE (Hartree)")
    for key in ("idxVB", "idxCB"):
        if key in params:
            _check(0 <= int(params[key]) < n_bands, f"{key} = {params[key]} is outside the {n_bands} fitted bands")
    params["nBands"] = n_bands

    out.mkdir(parents=True, exist_ok=True)
    cell_bohr = np.asarray(run["cell_angstrom"]) * ANGSTROM_TO_BOHR
    lines = ["scale = 1.0", "", "cell"] + [" ".join(f"{v:.10f}" for v in row) for row in cell_bohr] + ["", "atoms"]
    lines += [f"{s} " + " ".join(f"{v:.10f}" for v in pos) for s, pos in zip(run["symbols"], run["scaled_positions"])]
    (out / "system_0.par").write_text("\n".join(lines) + "\n")

    (out / "kpoints_0.par").write_text("".join(f"{k[0]:.10f} {k[1]:.10f} {k[2]:.10f} 1.0\n" for k in kpts[kidx]))
    # x: cumulative distance between the kept points, for the plot axis only
    sel = kpts[kidx]
    recip = 2 * np.pi * np.linalg.inv(np.asarray(run["cell_angstrom"])).T
    x = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(sel @ recip, axis=0), axis=1))])
    bands = E[kidx][:, first_band:first_band + n_bands] + energy_shift
    (out / "expBandStruct_0.par").write_text(
        "".join(f"{xi:.6f} " + " ".join(f"{e:.6f}" for e in row) + "\n" for xi, row in zip(x, bands)))
    (out / "bandWeights_0.par").write_text("".join(f"{float(w)}\n" for w in band_weights))
    (out / "input_0.par").write_text("".join(f"{k} = {v}\n" for k, v in params.items()))
    (out / "NN_config.par").write_text("".join(f"{k} = {v}\n" for k, v in cfg.items()))
    for el, p in init_params.items():
        (out / f"init_{el}Params.par").write_text("".join(f"{float(v)}\n" for v in p))
    labels = dict(run["high_symmetry"])
    summary = {
        "out_dir": str(out.resolve()),
        "files": sorted(p.name for p in out.iterdir()),
        "n_kpoints": len(kidx),
        "kpoint_labels": [labels.get(i, "") for i in kidx],
        "dft_bands_fitted": [first_band, first_band + n_bands - 1],
        "dft_bands_dropped_below": first_band,
        "energy_range_eV": [float(bands.min()), float(bands.max())],
    }
    (out / "bundle_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _par(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0]
        if "=" in line:
            k, v = (s.strip() for s in line.split("=", 1))
            out[k] = v
    return out


def read_fit(results_dir: str | Path, inputs_dir: str | Path) -> dict:
    """How well the trained pseudopotential reproduces the DFT bands: the
    last epoch_<N>_BS_sys0.dat in results_dir against inputs_dir's
    expBandStruct_0.par. RMS error (eV) over all bands and per band, the
    reference and fitted gap (idxVB/idxCB from input_0.par), and the files a
    reader should look at."""
    res, inp = Path(results_dir), Path(inputs_dir)
    files = sorted(res.glob("epoch_*_BS_sys0.dat"), key=lambda p: int(p.name.split("_")[1]))
    if not files:
        raise FileNotFoundError(f"no epoch_*_BS_sys0.dat in {res}: training did not reach a band-structure evaluation")
    fit = np.loadtxt(files[-1])
    ref = np.loadtxt(inp / "expBandStruct_0.par")
    _check(fit.shape == ref.shape, f"{files[-1].name} is {fit.shape}, the reference {ref.shape}")
    weights = np.loadtxt(inp / "bandWeights_0.par").reshape(-1)
    err = fit[:, 1:] - ref[:, 1:]
    params = _par(inp / "input_0.par")
    out = {
        "epoch_file": str(files[-1]),
        "epochs_evaluated": [int(p.name.split("_")[1]) for p in files],
        "rms_error_eV": float(np.sqrt(np.mean(err ** 2))),
        "weighted_rms_error_eV": float(np.sqrt(np.sum(weights * np.mean(err ** 2, axis=0)) / np.sum(weights))),
        "max_abs_error_eV": float(np.abs(err).max()),
        "rms_error_per_band_eV": [round(float(v), 4) for v in np.sqrt(np.mean(err ** 2, axis=0))],
    }
    if "idxVB" in params and "idxCB" in params:
        vb, cb = int(params["idxVB"]) + 1, int(params["idxCB"]) + 1  # +1: column 0 is the path coordinate
        out["gap_reference_eV"] = float(ref[:, cb].min() - ref[:, vb].max())
        out["gap_fit_eV"] = float(fit[:, cb].min() - fit[:, vb].max())
    cost = res / "training_cost.dat"
    if cost.exists():
        c = np.loadtxt(cost)
        c = c[:, -1] if c.ndim > 1 else c
        out["training_cost_first_last"] = [float(c[0]), float(c[-1])]
    out["plots"] = sorted(str(p) for p in res.glob("*plotBS.png"))[-2:] + sorted(str(p) for p in res.glob("final_plotBS.pdf"))
    out["final_pot_files"] = sorted(str(p) for p in res.glob("final_pot*.dat"))
    return out
