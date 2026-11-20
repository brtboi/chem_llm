"""SCF k-mesh convergence: finer and finer Monkhorst-Pack meshes until the
total energy stops moving.

The meshes come from scf_kmesh (even spacing along every reciprocal vector),
starting at a spacing and multiplying it by a factor < 1 each refinement,
skipping spacings that give a mesh already tried. A mesh is converged when
the next `confirmations` refinements each change the total energy per atom
by less than the threshold relative to it -- one small step can be luck, two
in a row is a plateau. The converged mesh is the coarsest such one.

    from chem_llm.dft.kmesh import mesh_sequence, converged_index, with_kmesh

The pw.x runs themselves are made by the converge_kmesh tool
(chem_llm.tools), which uses these to write each input and pick the result.
"""
from __future__ import annotations

import re
from pathlib import Path

RY_TO_EV = 13.605693122994

__all__ = ["mesh_sequence", "converged_index", "with_kmesh", "read_kmesh", "RY_TO_EV"]


def mesh_sequence(cell, start_spacing: float, factor: float, count: int,
                  min_spacing: float = 0.03) -> list[tuple[float, list[int]]]:
    """Up to `count` distinct (spacing, mesh) pairs, coarsest first. Each mesh
    is scf_kmesh(cell, spacing=...) at a spacing start * factor**n; spacings
    that reproduce the previous mesh are skipped, and every mesh is at least
    as dense as the one before along every axis."""
    from ase import Atoms

    from .kpath import scf_kmesh

    atoms = Atoms(cell=cell, pbc=True)
    out: list[tuple[float, list[int]]] = []
    spacing = float(start_spacing)
    while len(out) < count and spacing >= min_spacing:
        mesh = scf_kmesh(atoms, spacing=spacing)
        if not out or mesh != out[-1][1]:
            out.append((round(spacing, 6), mesh))
        spacing *= factor
    return out


def converged_index(energies_eV_per_atom: list[float], threshold_eV: float, confirmations: int = 2) -> int | None:
    """Index of the coarsest mesh whose energy is within threshold_eV of each
    of the next `confirmations` meshes' energies; None if there is none yet."""
    e = list(energies_eV_per_atom)
    for i in range(len(e) - confirmations):
        if all(abs(e[i + j] - e[i]) < threshold_eV for j in range(1, confirmations + 1)):
            return i
    return None


_KPOINTS_AUTO = re.compile(r"(?im)^\s*K_POINTS\s*[({]?\s*automatic\s*[)}]?\s*\n\s*([\d\s]+?)\s*$")


def read_kmesh(pw_in: str) -> tuple[list[int], list[int]] | None:
    """(mesh, shift) of an automatic K_POINTS card, or None."""
    m = _KPOINTS_AUTO.search(pw_in)
    if not m:
        return None
    nums = [int(v) for v in m.group(1).split()]
    if len(nums) != 6:
        return None
    return nums[:3], nums[3:]


def _set_namelist_var(text: str, name: str, value: str) -> str:
    """Set `name = value` in &CONTROL (replacing it if present)."""
    pattern = re.compile(rf"(?im)^(\s*){name}\s*=.*$")
    if pattern.search(text):
        return pattern.sub(lambda m: f"{m.group(1)}{name} = {value}", text, count=1)
    return re.sub(r"(?im)^(\s*&control\s*)$", lambda m: f"{m.group(1)}\n   {name} = {value}", text, count=1)


def with_kmesh(pw_in: str, mesh: list[int], shift: list[int] | None = None, *,
               outdir: str | None = None, pseudo_dir: str | None = None) -> str:
    """pw_in with its K_POINTS replaced by an automatic mesh (keeping the
    existing shift unless one is given), and optionally outdir / pseudo_dir
    set. Raises ValueError if the input has no automatic K_POINTS card."""
    found = read_kmesh(pw_in)
    if found is None:
        raise ValueError("pw.in has no 'K_POINTS automatic' card (the SCF needs a Monkhorst-Pack mesh)")
    shift = shift if shift is not None else found[1]
    card = f"K_POINTS automatic\n{mesh[0]} {mesh[1]} {mesh[2]}  {shift[0]} {shift[1]} {shift[2]}"
    text = _KPOINTS_AUTO.sub(card, pw_in, count=1)
    if outdir is not None:
        text = _set_namelist_var(text, "outdir", f"'{outdir}'")
    if pseudo_dir is not None:
        text = _set_namelist_var(text, "pseudo_dir", f"'{pseudo_dir}'")
    return text


def pseudo_dir_of(pw_in: str, directory: Path) -> Path | None:
    """The pseudo_dir pw.in names, resolved from the directory pw.x runs in;
    None when it names none (pw.x then uses $ESPRESSO_PSEUDO or its default,
    so a copy run elsewhere must not get one either)."""
    m = re.search(r"(?im)^\s*pseudo_dir\s*=\s*['\"]([^'\"]*)['\"]", pw_in)
    return (Path(directory) / m.group(1)).resolve() if m else None
