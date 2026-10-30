"""Guided setup for `chem-llm` with no arguments (or `chem-llm -i`).

1. Compound: the formula, then the structure -- a space group symbol, its
   number, or a phase name ("rutile"), which the agent resolves against
   Materials Project. Or search Materials Project here and pick from its
   structures for the formula (observed in experiment first).
2. Model: Qwen3.8-27B (offered for download if it is not in the Hugging Face
   cache), any other chat model in the cache, or Claude (bills; asks first).
3. Serving, for a local model: in-process with Hugging Face transformers, or
   a vLLM server -- reused if one is up, else started after the last screen.
   No vLLM installed -> chem-llm stops with an error.
4. Work dir, output, and optionally max steps / ensemble size.

It fills in the same argparse namespace the flags do, so the run is
identical, and prints the equivalent command line before starting.

Menus are numbered rather than arrow-key driven, so they work in any
terminal, including over ssh. Every prompt reads from `stream` when one is
given (tests feed scripted answers that way).
"""
from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from . import local_models
from .polymorphs import classify_spacegroup, lookup_polymorphs, parse_spacegroup
from .settings import Settings, normalize_backend

MAX_LISTED = 12


class Cancelled(Exception):
    pass


class SetupError(Exception):
    """Ends the wizard with an error (e.g. vLLM chosen but not installed)."""


def equivalent_command(args: argparse.Namespace) -> str:
    parts = ["chem-llm", args.compound]
    if args.spacegroup:
        parts.append(args.spacegroup)
        if args.spacegroup_number:
            parts += ["--spacegroup-number", str(args.spacegroup_number)]
    elif args.spacegroup_number:
        parts.append(str(args.spacegroup_number))
    if args.name:
        parts += ["--name", args.name] if (args.spacegroup or args.spacegroup_number) else [args.name]
    if args.model:
        parts += ["--model", args.model]
    if args.backend:
        parts += ["--backend", args.backend]
    if getattr(args, "start_vllm", False):
        parts.append("--start-vllm")
    if args.work_dir:
        parts += ["--work-dir", str(args.work_dir)]
    if args.keep:
        parts.append("--keep")
    if args.max_steps is not None:
        parts += ["--max-steps", str(args.max_steps)]
    if args.structures != 51:
        parts += ["--structures", str(args.structures)]
    for flag in ("pretty_progress", "summarize", "quiet"):
        if getattr(args, flag):
            parts.append("--" + flag.replace("_", "-"))
    return " ".join(shlex.quote(p) for p in parts)


class Wizard:
    def __init__(self, args, settings: Settings, console: Console | None = None, stream=None,
                 lookup=lookup_polymorphs, models=None, served=None, find_vllm=None, downloader=None):
        from . import vllm_server

        self.args, self.settings = args, settings
        self.console = console or Console()
        self.stream, self.lookup = stream, lookup
        self.models = models if models is not None else local_models.local_chat_models(settings)
        self.served = served or (lambda: vllm_server.served_at(args.vllm_url or settings.model.vllm_url))
        self.find_vllm = find_vllm or (lambda: vllm_server.find_vllm(settings))
        self.downloader = downloader or (lambda model: local_models.download(settings, model))

    # --- prompt helpers (all read from self.stream when set)
    def ask(self, text, **kw):
        return Prompt.ask(text, console=self.console, stream=self.stream, **kw)

    def ask_int(self, text, **kw):
        return IntPrompt.ask(text, console=self.console, stream=self.stream, **kw)

    def confirm(self, text, default=True):
        return Confirm.ask(text, console=self.console, stream=self.stream, default=default)

    def choose(self, options: list[str], default: int = 1) -> int:
        """Numbered menu; returns the 0-based index chosen."""
        for i, label in enumerate(options, 1):
            self.console.print(f"  [bold]{i}[/bold]  {label}")
        return self.ask_int("Choice", default=default, choices=[str(i) for i in range(1, len(options) + 1)],
                            show_choices=False) - 1

    def screen(self, step: int, total: int, title: str):
        self.console.print()
        self.console.rule(f"[bold]Step {step} of {total}: {title}", align="left")

    # --- 1. compound and structure
    def compound(self):
        from pymatgen.core import Composition

        while True:
            text = (self.args.compound or self.ask("Chemical formula (e.g. TiO2, GaAs, CsPbBr3)")).strip()
            self.args.compound = None
            try:
                Composition(text)
                if text:
                    self.args.compound = text
                    return
            except Exception:
                pass
            self.console.print(f"[red]{text!r} is not a chemical formula.[/red]")

    def structure(self):
        a = self.args
        if a.spacegroup or a.spacegroup_number or a.name:
            return  # given on the command line
        self.console.print("Space group as a symbol ([bold]P4_2/mnm[/bold]) or number ([bold]136[/bold]), or a "
                           "phase name ([bold]rutile[/bold]) for the agent to find on Materials Project.")
        while True:
            text = self.ask("Structure, or [bold]s[/bold] to search Materials Project for "
                            f"{a.compound}").strip()
            if not text:
                continue
            if text.lower() in ("s", "search"):
                if self.search():
                    break
                continue
            try:
                symbol, number, name = classify_spacegroup(text)
            except ValueError as e:
                self.console.print(f"[red]{e}[/red]")
                continue
            if name:
                if any(c.isdigit() for c in name) or "/" in name:
                    if not self.confirm(f"{name!r} is not a space group symbol pymatgen recognizes (Materials "
                                        "Project format: P4_2/mnm, Fd-3m). Pass it to the agent as a phase "
                                        "name anyway?", default=False):
                        continue
                self.console.print(f"The agent will list Materials Project's structures for {a.compound} and "
                                   f"pick the {name} phase.")
                a.name = name
                return
            a.spacegroup, a.spacegroup_number = symbol, number
            self.console.print(f"{symbol} (#{number})")
            break
        a.name = self.ask("Common name, used in the prompt (optional, e.g. rutile)", default="",
                          show_default=False).strip()

    def search(self) -> bool:
        """Pick from Materials Project's structures. False = back to typing one."""
        a = self.args
        self.console.print(f"Looking up {a.compound} on Materials Project ...")
        try:
            found = self.lookup(a.compound, self.settings.mp_api_key)
        except Exception as e:
            self.console.print(f"[yellow]Materials Project lookup failed ({type(e).__name__}: {e}).[/yellow]")
            return False
        if not found:
            self.console.print(f"[yellow]Materials Project has no structures for {a.compound}; the run fetches "
                               "its structure from there, so check the formula.[/yellow]")
            return False
        shown = found[:MAX_LISTED]
        table = Table(box=None, pad_edge=False)
        for col in ("", "space group", "#", "crystal system", "", "MP id", "atoms", "E above hull", "MP gap"):
            table.add_column(col, justify="right" if col in ("", "#", "atoms") else "left")
        for i, e in enumerate(shown, 1):
            ehull, gap = e["energy_above_hull_eV_per_atom"], e["band_gap_eV"]
            table.add_row(str(i), e["spacegroup_symbol"], str(e["spacegroup_number"]), e["crystal_system"],
                          "observed" if e["observed"] else "[dim]hypothetical[/dim]", e["material_id"],
                          str(e["nsites"]), "stable" if not ehull else f"+{ehull:.3f} eV/atom",
                          "" if gap is None else ("metal" if gap < 1e-3 else f"{gap:.2f} eV"))
        table.add_row(str(len(shown) + 1), "back", "", "", "", "", "", "", "")
        self.console.print(table)
        if len(found) > len(shown):
            self.console.print(f"  ({len(found) - len(shown)} less stable space groups not listed; type one instead)")
        self.console.print("  observed: matches an experimental (ICSD) structure. MP gap is Materials Project's "
                           "PBE value, for orientation only.")
        i = self.ask_int("Choice", default=1, choices=[str(k) for k in range(1, len(shown) + 2)],
                         show_choices=False) - 1
        if i == len(shown):
            return False
        a.spacegroup, a.spacegroup_number = shown[i]["spacegroup_symbol"], shown[i]["spacegroup_number"]
        return True

    # --- 2. model, 3. serving
    def model(self) -> str:
        """Sets args.model / args.backend for Claude; returns 'local' or 'claude'."""
        a, default_id = self.args, self.settings.model.qwen_model or local_models.DEFAULT_MODEL
        cached = {m["id"]: m for m in self.models}
        ids = [default_id] + [m for m in cached if m != default_id]
        options = []
        for mid in ids:
            if mid in cached:
                tag = f"[green]downloaded[/green], {cached[mid]['gb']} GB"
            else:
                tag = f"[yellow]not downloaded[/yellow] (~{local_models.DEFAULT_MODEL_GB} GB)"
            note = "default, benchmarked" if mid == default_id else "[dim]not benchmarked with this agent[/dim]"
            options.append(f"{mid}  ({tag}; {note})")
        options.append(f"Claude ({self.settings.model.claude_model}, Anthropic API): "
                       "[bold red]bills your API credits every step[/bold red]")
        current = normalize_backend(a.backend or "")
        default = len(options) if current == "claude" else (ids.index(a.model) + 1 if a.model in ids else 1)
        while True:
            i = self.choose(options, default=default)
            if i == len(ids):
                if not self.confirm("The Claude backend bills your Anthropic API credits for every step (a run "
                                    "is 40-60 calls). Use it?", default=False):
                    continue
                if not self.settings.anthropic_api_key:
                    raise SetupError("Claude needs ANTHROPIC_API_KEY (environment, .env or config.yaml)")
                a.backend, a.model = "claude", None
                return "claude"
            mid = ids[i]
            if mid not in cached:
                free = local_models.free_gb(self.settings)
                where = self.settings.hf_cache_dir or "the Hugging Face cache"
                room = f"; {free:.0f} GB free there" if free is not None else ""
                if free is not None and free < local_models.DEFAULT_MODEL_GB * 1.1:
                    self.console.print(f"[red]Not enough disk space to download {mid} to {where}{room}.[/red]")
                    continue
                if not self.confirm(f"Download {mid} (~{local_models.DEFAULT_MODEL_GB} GB) to {where}{room} now?",
                                    default=True):
                    continue
                with self.console.status(f"downloading {mid} ..."):
                    self.downloader(mid)
                self.console.print(f"Downloaded {mid}.")
            a.model = None if mid == default_id else mid
            self._model_id = mid
            return "local"

    def serving(self):
        a, mid = self.args, self._model_id
        url = a.vllm_url or self.settings.model.vllm_url
        served = self.served()
        if served is not None and mid in served:
            vllm_line = f"[green]a server at {url} is already serving it[/green]"
        elif served is not None:
            vllm_line = f"[yellow]the server at {url} serves {', '.join(served)} instead[/yellow]"
        else:
            vllm_line = "starts a server for it after the last screen (~5 min), kept running for later runs"
        options = [f"vLLM server: ~5 s/step; {vllm_line}",
                   "Hugging Face transformers, in this process: ~50 s/step, loads the model each run"]
        current = normalize_backend(a.backend or self.settings.model.backend)
        i = self.choose(options, default=2 if current == "huggingface" and a.backend else 1)
        if i == 1:
            a.backend, a.start_vllm = "huggingface", False
            return
        if served is not None and mid not in served:
            raise SetupError(f"the vLLM server at {url} serves {served}, not {mid}: stop it, pick that model, "
                             "or point --vllm-url at another port")
        if served is None and self.find_vllm() is None:
            raise SetupError("vLLM not found. Install it in its own venv (it pins its own torch), e.g.\n"
                             "    python -m venv ~/venvs/vllm && ~/venvs/vllm/bin/pip install vllm\n"
                             "then set model.vllm_executable in config.yaml (or VLLM_VENV) to it, or choose "
                             "Hugging Face transformers.")
        a.backend, a.start_vllm = "vllm", served is None

    # --- 4. where to run, output
    def workdir(self):
        a = self.args
        default = str(a.work_dir or (self.settings.work_root / a.compound))
        a.work_dir = self.ask("Work directory", default=default)
        p = Path(a.work_dir)
        if p.is_dir() and any(p.iterdir()):
            a.keep = not self.confirm(f"{p} already has files. Clear it first (*.jsonl run logs are kept)?",
                                      default=True)

    def output(self):
        a = self.args
        i = self.choose(["pretty progress: one readable line per action",
                         "raw stream: every model output, tool call and result",
                         "quiet: print nothing while it runs"], default=1)
        a.pretty_progress, a.quiet = i == 0, i == 2
        a.summarize = self.confirm("Print the run summary when it finishes?", default=True)
        if not self.confirm("Change advanced settings (max steps, ensemble size)?", default=False):
            return
        a.max_steps = self.ask_int("Max agent steps", default=a.max_steps or self.settings.agent.max_steps)
        a.structures = self.ask_int("Perturbed structures in the ensemble", default=a.structures)
        self.console.print("  Temperature, history and the QE launch are set in config.yaml.")

    def run(self) -> argparse.Namespace:
        a = self.args
        self.console.rule("[bold]chem-llm: band structure with Quantum ESPRESSO")
        self.console.print("Flags skip this: chem-llm TiO2 P4_2/mnm --backend vllm ...  (chem-llm --help)")
        self.screen(1, 4, "compound and structure")
        self.compound()
        self.structure()
        self.screen(2, 4, "model")
        local = self.model() == "local"
        if local:
            self.console.print()
            self.console.print("[bold]How should it run?[/bold]")
            self.serving()
        self.screen(3, 4, "where to run")
        self.workdir()
        self.screen(4, 4, "output")
        self.output()
        self.console.print()
        self.console.rule("[bold]Ready", align="left")
        what = (f"{a.spacegroup} (#{a.spacegroup_number})" + (f", {a.name}" if a.name else "")
                if a.spacegroup else f"the {a.name} phase (the agent picks its space group)")
        self.console.print(f"{a.compound}, {what}  ->  {a.work_dir}")
        self.console.print("Equivalent command:")
        self.console.print("  " + equivalent_command(a), markup=False, highlight=False, soft_wrap=True)
        if not self.confirm("Start the run?", default=True):
            raise Cancelled
        return a


def wizard(args: argparse.Namespace, settings: Settings, **kw) -> argparse.Namespace | None:
    """Fill `args` interactively. None if the user cancels (n, Ctrl-C, Ctrl-D)
    or a choice cannot work (SetupError: printed, chem-llm exits non-zero)."""
    console = kw.get("console") or Console()
    kw["console"] = console
    try:
        return Wizard(args, settings, **kw).run()
    except SetupError as e:
        console.print(f"\n[red]chem-llm: {e}[/red]", highlight=False)
        return None
    except (Cancelled, KeyboardInterrupt, EOFError):
        console.print("\nCancelled.")
        return None
