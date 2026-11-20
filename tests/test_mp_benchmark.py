"""Workflow settings, SCF k-mesh convergence, and scoring against Materials
Project's band structure -- all offline (QE and MP are faked)."""
import json

import numpy as np
import pytest

from chem_llm.core.tasks import build_task
from chem_llm.dft.kmesh import converged_index, mesh_sequence, read_kmesh, with_kmesh
from chem_llm.dft.materials_project import compare_bands, mp_kpath, potcar_valence_note
from chem_llm.settings import Settings, WorkflowSettings, activated

RUTILE_CELL = [[4.65, 0, 0], [0, 4.65, 0], [0, 0, 2.97]]

PW_IN = """&CONTROL
   calculation = 'scf'
   outdir = './tmp'
   pseudo_dir = './'
/
&SYSTEM
   ibrav = 0
   nat = 1
   ntyp = 1
   ecutwfc = 40
/
&ELECTRONS
/
ATOMIC_SPECIES
Si 28.085 Si.upf

K_POINTS automatic
4 4 4  0 0 0

CELL_PARAMETERS angstrom
4.65 0.0 0.0
0.0 4.65 0.0
0.0 0.0 2.97

ATOMIC_POSITIONS crystal
Si 0.0 0.0 0.0
"""


# ---------------------------------------------------------------- settings
def test_mode_fills_unset_choices_and_overrides_win():
    agent = WorkflowSettings().resolved()
    assert (agent["kpath"], agent["pseudopotentials"], agent["spin_orbit"], agent["functional"]) == \
        ("seekpath", "agent", "agent", "pbesol")
    mp = WorkflowSettings(mode="match_mp").resolved()
    assert (mp["kpath"], mp["pseudopotentials"], mp["spin_orbit"], mp["functional"]) == \
        ("materials_project", "match_mp", "off", "pbe")
    mixed = WorkflowSettings(mode="match_mp", kpath="seekpath", spin_orbit=True).resolved()
    assert mixed["kpath"] == "seekpath" and mixed["spin_orbit"] == "on"  # yaml `on` reads as True
    assert mixed["kmesh"]["threshold_meV_per_atom"] == 1.0 and mixed["kmesh"]["start_spacing"] == 0.25


def test_bad_workflow_value_is_refused():
    with pytest.raises(ValueError, match="workflow.kpath"):
        WorkflowSettings(kpath="vasp")


def test_workflow_section_is_read_from_yaml():
    s = Settings.from_dict({"workflow": {"mode": "match_mp", "kmesh_threshold_meV_per_atom": 0.5}})
    assert s.workflow.mode == "match_mp" and s.workflow.kmesh_threshold_meV_per_atom == 0.5


def test_task_text_follows_the_workflow():
    agent = build_task("TiO2", "P4_2/mnm", workflow=WorkflowSettings(kmesh_convergence=False).resolved())
    assert agent == build_task("TiO2", "P4_2/mnm")  # agent mode, no convergence: the original task
    conv = build_task("TiO2", "P4_2/mnm", workflow=WorkflowSettings().resolved())
    assert "call converge_kmesh on calculations/000" in conv and "1.0 meV/atom" in conv
    mp = build_task("TiO2", "P4_2/mnm", functional="pbe", workflow=WorkflowSettings(mode="match_mp").resolved())
    assert "load_kpath('structures/kpath.json')" in mp
    assert "PAW (kind 'paw'), generator 'pbe'" in mp
    assert "spin-orbit is OFF" in mp and "decide the relativity" not in mp


# ---------------------------------------------------------------- k-mesh
def test_mesh_sequence_refines_every_axis_and_skips_repeats():
    seq = mesh_sequence(RUTILE_CELL, 0.25, 0.8, 6)
    meshes = [m for _, m in seq]
    assert meshes[0] == [6, 6, 9]  # rutile's short c gets more points
    assert len({tuple(m) for m in meshes}) == len(meshes)
    assert all(all(b >= a for a, b in zip(m0, m1)) for m0, m1 in zip(meshes, meshes[1:]))


def test_converged_index_needs_every_confirmation():
    thr = 0.001
    assert converged_index([0.0, -0.01, -0.0105, -0.0106], thr, 2) == 1
    assert converged_index([0.0, -0.01, -0.0105, -0.013], thr, 2) is None  # second refinement moved
    assert converged_index([0.0, -0.0001], thr, 2) is None  # not enough meshes yet


def test_with_kmesh_rewrites_only_what_it_should():
    text = with_kmesh(PW_IN, [8, 8, 11], outdir="./tmp2", pseudo_dir="/abs/pseudo")
    assert read_kmesh(text) == ([8, 8, 11], [0, 0, 0])
    assert "outdir = './tmp2'" in text and "pseudo_dir = '/abs/pseudo'" in text
    assert "ecutwfc = 40" in text
    with pytest.raises(ValueError):
        with_kmesh("K_POINTS gamma\n", [2, 2, 2])


def test_converge_kmesh_tool(tmp_path, monkeypatch):
    """Energies that settle after the third mesh: the tool picks it, rewrites pw.in, saves the table."""
    from chem_llm import tools

    calc = tmp_path / "calculations" / "000"
    calc.mkdir(parents=True)
    (calc / "pw.in").write_text(PW_IN)
    energies_Ry = iter([-10.0, -10.01, -10.0102, -10.01025, -10.01026, -10.01026])

    def fake_launch(work_dir, steps):
        mesh = read_kmesh((work_dir / "pw.in").read_text())[0]
        (work_dir / "pw.out").write_text(
            f"number of atoms/cell = 1\nnumber of k points= {mesh[0] * mesh[1]}\n"
            f"!    total energy              = {next(energies_Ry):.8f} Ry\n"
            "convergence has been achieved in 5 iterations\nJOB DONE.\n")
        return {"success": True, "timings_s": {"scf": 1.0}}

    monkeypatch.setattr(tools, "_launch_qe", fake_launch)
    with activated(Settings(work_root=tmp_path)):
        r = tools.converge_kmesh(str(calc))
    assert r["success"] and r["converged"]
    # -10.0102 -> -10.01025 -> -10.01026 Ry: within 1 meV/atom (7.3e-5 Ry) of both refinements
    assert len(r["meshes"]) == 5 and r["mesh"] == r["meshes"][2]["mesh"]
    assert read_kmesh((calc / "pw.in").read_text())[0] == r["mesh"]
    saved = json.loads((calc / "kmesh_convergence.json").read_text())
    assert saved["mesh"] == r["mesh"] and saved["threshold_meV_per_atom"] == 1.0
    assert "spacing=" in r["note"]


# ---------------------------------------------------------------- Materials Project
def _fake_reference(nk_per_branch=11, gap=1.0):
    """A two-branch path G-X | Y-G on the rutile cell, with simple cosine bands:
    3 valence bands below 0, 2 conduction bands above `gap`."""
    g, x, y = np.zeros(3), np.array([0.5, 0, 0]), np.array([0, 0.5, 0])
    k1 = [g + (x - g) * t for t in np.linspace(0, 1, nk_per_branch)]
    k2 = [y + (g - y) * t for t in np.linspace(0, 1, nk_per_branch)]
    k = np.array(k1 + k2)
    phase = np.cos(2 * np.pi * np.linalg.norm(k, axis=1))
    bands = np.stack([-6 + phase, -3 + 0.5 * phase, -0.5 * (1 - phase), gap + 0.5 * (1 - phase), gap + 3 + phase],
                     axis=1)
    labels = [""] * len(k)
    labels[0], labels[nk_per_branch - 1], labels[nk_per_branch], labels[-1] = "\\Gamma", "X", "Y", "\\Gamma"
    return {
        "material_id": "mp-test", "path_type": "setyawan_curtarolo", "cell": RUTILE_CELL,
        "symbols": ["Si"], "scaled_positions": [[0, 0, 0]], "kpoints": k.tolist(),
        "labels": ["GAMMA" if lab == "\\Gamma" else lab for lab in labels],
        "branches": [{"start": 0, "end": nk_per_branch - 1, "name": "GAMMA-X"},
                     {"start": nk_per_branch, "end": 2 * nk_per_branch - 1, "name": "Y-GAMMA"}],
        "energies": {"up": bands.tolist()}, "efermi": 0.1, "is_metal": False, "is_spin_polarized": False,
        "band_gap_eV": gap, "vbm_eV": 0.0, "cbm_eV": gap, "vbm_band": 2, "calculation": {},
    }


def test_mp_kpath_has_a_break_and_no_distance_across_it():
    ref = _fake_reference()
    kp = mp_kpath(ref)
    assert kp["path"] == "GAMMA-X | Y-GAMMA"
    assert kp["segments"] == [(0, 11), (11, 22)]
    assert kp["x"][11] == kp["x"][10]  # X|Y discontinuity adds no length
    assert [t["label"] for t in kp["ticks"]] == ["GAMMA", "X|Y", "GAMMA"]


def test_compare_bands_same_path_with_extra_semicore_band_and_shift():
    ref = _fake_reference()
    R = np.asarray(ref["energies"]["up"])
    # the run has one extra deep semicore band and its energies sit 5 eV higher
    run = np.hstack([np.full((len(R), 1), -40.0), R + 5.0])
    r = compare_bands(mp_kpath(ref), run, ref, n_occupied=4)
    assert r["k_matching"] == "same path" and r["band_offset"] == 1
    assert r["rms_eV"] == pytest.approx(0, abs=1e-9) and r["gap_error_eV"] == pytest.approx(0, abs=1e-9)


def test_compare_bands_interpolates_a_denser_path_and_sees_a_real_error():
    ref = _fake_reference(nk_per_branch=11)
    dense = _fake_reference(nk_per_branch=41)   # the same path at 4x the density
    E = np.asarray(dense["energies"]["up"]).copy()
    E[:, 3:] += 0.1                              # conduction bands 0.1 eV too high
    r = compare_bands(mp_kpath(dense), E, ref, n_occupied=3)
    assert r["n_kpoints"] == len(ref["kpoints"]) and r["k_matching"].startswith("interpolated")
    assert r["gap_error_eV"] == pytest.approx(0.1, abs=1e-6)
    assert r["per_band_rms_eV"]["1"] == pytest.approx(0.1, abs=0.02)   # first conduction band
    assert r["per_band_rms_eV"]["0"] < 0.02                              # top valence band


def test_potcar_names_say_which_semicore_states():
    assert "semicore p" in potcar_valence_note("PAW_PBE Ti_pv 07Sep2000")["meaning"]
    assert potcar_valence_note("PAW_PBE O 08Apr2002")["meaning"].startswith("standard")


# ---------------------------------------------------------------- benchmark checks
def test_mp_checks_are_soft_unless_the_suite_sets_tolerances():
    from chem_llm.bench import mp_checks, run_scores

    mp = {"rms_eV": 0.2, "max_abs_eV": 0.5, "gap_error_eV": 0.05, "run_gap_eV": 1.83, "mp_gap_eV": 1.78,
          "detail": "RMS 0.200 eV", "kmesh": {"mesh": [6, 6, 9], "converged": True, "n_scf": 4}}
    soft = mp_checks(mp, None)
    assert [c["name"] for c in soft] == ["mp_band_rms", "mp_gap"] and not any(c["hard"] for c in soft)
    hard = mp_checks(mp, {"rms_eV": 0.15, "gap_eV": 0.15})
    assert [c["ok"] for c in hard] == [False, True] and all(c["hard"] for c in hard)
    assert mp_checks(None, {"rms_eV": 0.15})[0]["ok"] is False
    scores = run_scores({"band_gap": 1.83}, mp)
    assert scores["mp_band_rms_eV"] == 0.2 and scores["kmesh_points"] == 324 and scores["kmesh_scf_runs"] == 4


def test_mp_suite_runs_in_its_own_work_root():
    from chem_llm.bench import BENCH, load_suites, work_dir_of

    suites, cases = load_suites()
    assert work_dir_of("gaas", cases["gaas"], suites["mp"]) == BENCH / "work" / "mp" / "gaas"
    assert "sandbox" in str(work_dir_of("gaas", cases["gaas"]))  # the core suite's own dir is unchanged
    assert suites["mp"]["workflow"] == {"mode": "match_mp"}


def test_sidecar_is_found_from_a_directory_or_a_structure(tmp_path):
    from chem_llm.dft.materials_project import sidecar

    (tmp_path / "structures").mkdir()
    (tmp_path / "structures" / "materials_project.json").write_text('{"material_id": "mp-149"}')
    assert sidecar(tmp_path / "structures")["material_id"] == "mp-149"
    assert sidecar(tmp_path / "structures" / "structure_000.cif")["material_id"] == "mp-149"
    assert sidecar(tmp_path) == {}


def test_kmesh_copies_keep_a_missing_pseudo_dir_missing(tmp_path):
    from chem_llm.dft.kmesh import pseudo_dir_of

    no_dir = PW_IN.replace("   pseudo_dir = './'\n", "")
    assert pseudo_dir_of(no_dir, tmp_path) is None   # pw.x would use $ESPRESSO_PSEUDO: so must the copies
    assert pseudo_dir_of(PW_IN, tmp_path) == tmp_path.resolve()
