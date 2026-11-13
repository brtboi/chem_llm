"""Reading a finished chem_llm QE run and writing a DeePseudopot input bundle.

The bookkeeping between the two formats is deterministic, so it lives here
rather than in agent-written code (the same split as chem_llm.dft.kpath /
qe_output for the DFT agent). The physics decisions -- which DFT bands the
empirical pseudopotential should reproduce, which k-points, weights, basis
cutoff, training length -- stay with the agent.

    from chem_llm.dpp.io import read_qe_run, pseudo_orbitals, write_bundle, read_fit

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

    from ..dft.kpath import _read_filband, load_kpath
    from ..dft.qe_output import band_edges, read_pw_out

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


def _upf_radial(upf_path: str | Path) -> dict:
    """The radial mesh, local potential (Hartree) and pseudo-orbitals of a UPF v2 file."""
    text = Path(upf_path).read_text(errors="replace")

    def array(tag):
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", text, re.S)
        if not m:
            raise ValueError(f"{upf_path}: no <{tag}> (only UPF v2 files are supported)")
        return np.array(m.group(1).split(), float)

    z = float(re.search(r'z_valence\s*=\s*"\s*([-+\d.EeDd]+)', text).group(1).replace("D", "E"))
    orbitals = []
    for m in re.finditer(r"<PP_CHI\.\d+([^>]*)>(.*?)</PP_CHI\.\d+>", text, re.S):
        a = dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', m.group(1)))
        orbitals.append((a.get("label", "?").upper(), float(a.get("occupation", "0")), np.array(m.group(2).split(), float)))
    return {"z": z, "r": array("PP_R"), "rab": array("PP_RAB"), "vloc": array("PP_LOCAL") / 2.0, "orbitals": orbitals}


def _radial_ft(q, r, rab, f_r2):
    """4 pi * integral f(r) r^2 sin(qr)/(qr) dr, given f(r) r^2 on the UPF mesh."""
    return np.array([4 * np.pi * np.sum(f_r2 * np.sinc(qi * r / np.pi) * rab) for qi in q])


def screened_form_factors(run: dict, semicore: dict, q_max: float = 30.0, n_q: int = 4097) -> dict:
    """Screened atomic form factors v(q) (Hartree * Bohr^3, DeePseudopot's
    convention: V(G) = sum_atoms e^{iG.tau} v(|G|) / cell volume) from the
    run's own UPF files -- a physical starting point for the fit.

    semicore    {element: [orbital labels]} dropped from the fit, e.g.
                {"Ti": ["3S", "3P"]}: their charge (the UPF pseudo-orbitals)
                is added to the ion, so Ti becomes an effective +4 ion.
    The ions are then screened by the remaining valence electrons, the ones
    in the fitted bands, as a uniform electron gas (Lindhard dielectric
    function at the cell's valence density).
    """
    from scipy.special import erf

    q = np.linspace(0.0, q_max, n_q)
    volume = abs(np.linalg.det(np.asarray(run["cell_angstrom"]))) * ANGSTROM_TO_BOHR ** 3
    species = list(dict.fromkeys(run["symbols"]))
    _check(set(semicore) <= set(species), f"semicore names elements not in the cell: {sorted(set(semicore) - set(species))}")
    numerators, z_eff, n_semicore = {}, {}, 0.0
    for el in species:
        u = _upf_radial(run["pseudopotentials"][el])
        r, rab, z = u["r"], u["rab"], u["z"]
        labels = {o["label"] for o in pseudo_orbitals(run["pseudopotentials"][el])}
        wanted = {s.upper() for s in semicore.get(el, [])}
        _check(wanted <= labels, f"{el}: semicore {sorted(wanted - labels)} not among the UPF orbitals {sorted(labels)}")
        # local potential with its -z/r tail removed (transformed numerically) and added back analytically
        safe_r = np.where(r > 0, r, 1.0)
        short = np.where(r > 0, u["vloc"] + z * erf(r) / safe_r, u["vloc"] + 2 * z / np.sqrt(np.pi))
        rho_r2 = sum((occ * chi ** 2 / (4 * np.pi) for label, occ, chi in u["orbitals"] if label in wanted),
                     np.zeros_like(r))
        n_sc = sum(occ for label, occ, _ in u["orbitals"] if label in wanted)
        # q^2 * v_ion(q): finite at q = 0, where it is -4 pi (z - n_sc)
        numerators[el] = (q ** 2 * _radial_ft(q, r, rab, short * r ** 2) - 4 * np.pi * z * np.exp(-q ** 2 / 4)
                          + 4 * np.pi * _radial_ft(q, r, rab, rho_r2))
        z_eff[el] = z - n_sc
        n_semicore += n_sc * run["symbols"].count(el)
    n_screen = run["n_electrons"] - n_semicore
    kf = (3 * np.pi ** 2 * n_screen / volume) ** (1 / 3)
    x = q / (2 * kf)
    with np.errstate(divide="ignore", invalid="ignore"):
        lindhard = 0.5 + (1 - x ** 2) / (4 * x) * np.log(np.abs((1 + x) / (1 - x)))
    lindhard[x == 0] = 1.0
    lindhard[np.isclose(x, 1.0)] = 0.5
    denominator = q ** 2 + 4 * kf / np.pi * lindhard
    return {"q": q, "v": {el: numerators[el] / denominator for el in species}, "z_eff": z_eff,
            "n_screening_electrons": n_screen, "cell_volume_bohr3": volume}


def zunger_form(params, q):
    """DeePseudopot's local form: a0 (q^2 - a1) / (a2 exp(a3 q^2) - 1)."""
    a0, a1, a2, a3 = params[:4]
    return a0 * (np.asarray(q) ** 2 - a1) / (a2 * np.exp(a3 * np.asarray(q) ** 2) - 1)


def initial_zunger_params(run: dict, semicore: dict, q_fit_max: float = 8.0) -> dict:
    """Zunger-form starting parameters for every element, least-squares fitted
    to screened_form_factors over 0 <= q <= q_fit_max (1/Bohr). Returns
    {element: {"params": [9 floats] (non-local and SOC terms 0),
    "fit_rms": ..., "v0": v(0) of the screened potential, "z_eff": ...}}
    (fit_rms and v0 in Hartree * Bohr^3, the form factor's unit -- not eV);
    pass {el: d["params"]} to write_bundle's init_params."""
    from scipy.optimize import least_squares

    ff = screened_form_factors(run, semicore)
    q = ff["q"]
    m = q <= q_fit_max
    out = {}
    for el, v in ff["v"].items():
        fit = least_squares(lambda p: zunger_form(p, q[m]) - v[m], x0=[abs(v[0]) / 100, 5.0, 1.5, 0.3],
                            bounds=([-np.inf, 0.0, 1.0001, 1e-3], [np.inf, 200.0, 50.0, 5.0]))
        out[el] = {"params": [float(p) for p in fit.x] + [0.0] * 5,
                   "fit_rms": float(np.sqrt(np.mean(fit.fun ** 2))), "v0": float(v[0]), "z_eff": ff["z_eff"][el]}
    return out


def best_zunger_params(results_dir: str | Path) -> dict:
    """The lowest-cost Zunger parameters a Monte Carlo stage
    (train_deepseudopot mode='zunger_mc') found: {element: [9 floats]},
    from its best_<element>Params.dat files -- the init_params of the next stage."""
    res = Path(results_dir)
    files = sorted(res.glob("best_*Params.dat"))
    if not files:
        raise FileNotFoundError(f"no best_<element>Params.dat in {res}: the Monte Carlo stage found no improvement "
                                "or did not run")
    return {p.name[len("best_"):-len("Params.dat")]: [float(v) for v in p.read_text().split()] for p in files}


def _check(cond, msg):
    if not cond:
        raise ValueError(msg)


def write_bundle(out_dir: str | Path, run: dict, *, first_band: int, n_bands: int, kpoint_indices,
                 band_weights, input_params: dict, nn_config: dict, init_params: dict,
                 energy_shift: float = 0.0, mc_opts: dict | None = None) -> dict:
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
    mc_opts         for a Monte Carlo stage over the Zunger parameters
                    (train_deepseudopot mode='zunger_mc'): mcOpts1.par keys,
                    betas [cold, hot], stepsAtTemp [cold, hot], tempStepSizeMod
                    [cold, hot], totalIter

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
    # main_mc_Zunger.py's cost skips reference energies that are exactly 0 as missing data
    # (fit_mc.calcIndivMSE: abs(E) > 1e-15) -- with the VBM shifted to 0 that would be the VBM.
    # As written to 6 decimals, 1e-6 eV keeps the point in the fit and changes nothing else.
    zeros = np.abs(bands) < 1e-6
    bands = np.where(zeros, 1e-6, bands)
    (out / "expBandStruct_0.par").write_text(
        "".join(f"{xi:.6f} " + " ".join(f"{e:.6f}" for e in row) + "\n" for xi, row in zip(x, bands)))
    (out / "bandWeights_0.par").write_text("".join(f"{float(w)}\n" for w in band_weights))
    (out / "input_0.par").write_text("".join(f"{k} = {v}\n" for k, v in params.items()))
    (out / "NN_config.par").write_text("".join(f"{k} = {v}\n" for k, v in cfg.items()))
    for el, p in init_params.items():
        (out / f"init_{el}Params.par").write_text("".join(f"{float(v)}\n" for v in p))
    if mc_opts is not None:
        opts = {"betas": [20, 2], "stepsAtTemp": [100, 30], "tempStepSizeMod": [1.0, 4.0], "totalIter": 3000,
                **mc_opts, "fitDefPot": False, "fitCoupling": False, "optGaps": False}
        for key in ("betas", "stepsAtTemp", "tempStepSizeMod"):
            _check(len(opts[key]) == 2, f"mc_opts[{key!r}] needs two values, [cold, hot]")
        # read_mc_opts wants 'key = a b', spaces only, no commas
        (out / "mcOpts1.par").write_text("".join(
            f"{k} = {' '.join(str(x) for x in v) if isinstance(v, (list, tuple)) else v}\n" for k, v in opts.items()))
    labels = dict(run["high_symmetry"])
    summary = {
        "out_dir": str(out.resolve()),
        "files": sorted(p.name for p in out.iterdir()),
        "n_kpoints": len(kidx),
        "kpoint_labels": [labels.get(i, "") for i in kidx],
        "dft_bands_fitted": [first_band, first_band + n_bands - 1],
        "dft_bands_dropped_below": first_band,
        "energy_range_eV": [float(bands.min()), float(bands.max())],
        "zero_energies_moved_to_1e-6": int(zeros.sum()),
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
    """How well the trained pseudopotential reproduces the DFT bands, against
    inputs_dir's expBandStruct_0.par: for gradient training the last
    epoch_<N>_BS_sys0.dat in results_dir, for a Monte Carlo stage over the
    Zunger parameters its best band structure (bestBandStruct_<N>_0.dat).
    RMS error (eV) over all bands and per band, the reference and fitted gap
    (idxVB/idxCB from input_0.par), and the files a reader should look at."""
    res, inp = Path(results_dir), Path(inputs_dir)
    files = sorted(res.glob("epoch_*_BS_sys0.dat"), key=lambda p: int(p.name.split("_")[1]))
    stage = "gradient"
    if not files:
        files = sorted(res.glob("bestBandStruct_*_0.dat"), key=lambda p: int(p.name.split("_")[1]))
        stage = "zunger_mc"
    if not files:
        raise FileNotFoundError(f"no epoch_*_BS_sys0.dat or bestBandStruct_*_0.dat in {res}: training did not "
                                "reach a band-structure evaluation")
    fit = np.loadtxt(files[-1])
    ref = np.loadtxt(inp / "expBandStruct_0.par")
    _check(fit.shape == ref.shape, f"{files[-1].name} is {fit.shape}, the reference {ref.shape}")
    weights = np.loadtxt(inp / "bandWeights_0.par").reshape(-1)
    err = fit[:, 1:] - ref[:, 1:]
    params = _par(inp / "input_0.par")
    out = {
        "stage": stage,
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
    for name in ("final_training_cost.dat", "training_cost.dat"):
        if (res / name).exists():
            c = np.loadtxt(res / name)
            c = c[:, -1] if c.ndim > 1 else c
            out["training_cost_first_last"] = [float(c[0]), float(c[-1])]
            break
    if (res / "iterations.dat").exists():
        # header 'newMSE curMSE bstMSE %Accpt ...'; the first row lacks the leading
        # iteration number the others have, so the best cost is its 3rd column, their 4th
        rows = [line.split() for line in (res / "iterations.dat").read_text().splitlines()[1:] if line.strip()]
        best = [float(r[2]) for r in rows[:1]] + [float(r[3]) for r in rows[1:]]
        marks = sorted({0, *(len(best) * i // 4 for i in range(1, 4)), len(best) - 1})
        out["mc_iterations"] = len(best) - 1
        out["mc_best_cost_by_quarter"] = [round(float(best[i]), 3) for i in marks]
    out["plots"] = sorted(str(p) for p in res.glob("*plotBS.png"))[-2:] + sorted(str(p) for p in res.glob("final_plotBS.pdf"))
    out["final_pot_files"] = sorted(str(p) for p in res.glob("final_pot*.dat"))
    return out
