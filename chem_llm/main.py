"""Command-line entry point: `chem-llm <compound> <spacegroup>`.

    chem-llm TiO2 P4_2/mnm --name rutile --work-dir runs/tio2
    chem-llm TiO2 P4_2/mnm --pretty-print > summary.txt
    chem-llm C Fd-3m --model Qwen/Qwen3.8-27B --pw-x /opt/qe/bin/pw.x
    chem-llm --summarize runs/tio2          # summary of a past run (no GPU)
    chem-llm --build-docs

--pretty-print: when the run finishes, print a readable summary -- what the
agent did, its results and key choices in order, and where it struggled --
instead of the raw step-by-step stream, which goes to stderr. stdout then
holds only the summary.

A thin wrapper over DFTAgent for a single run; for a sweep, build one agent
in Python and call `run()` per compound so the model loads once.
"""
from __future__ import annotations

import argparse
import contextlib
import sys

from .agent import DFTAgent
from .settings import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chem-llm", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("compound", nargs="?", help="Chemical formula, e.g. TiO2.")
    parser.add_argument("spacegroup", nargs="?",
                        help="Space group in Materials Project format, e.g. P4_2/mnm, Fd-3m.")
    parser.add_argument("--spacegroup-number", type=int, default=None)
    parser.add_argument("--name", default="", help="Common name for the structure, e.g. rutile.")

    parser.add_argument("--config", default=None, help="Path to a config.yaml (default: search cwd and parents).")
    parser.add_argument("--work-dir", default=None, help="Where to run (default: <work_root>/<compound>).")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--keep", action="store_true", help="Keep existing files in the work dir.")

    parser.add_argument("--model", default=None, help="Local model id (default: from config, else Qwen3).")
    parser.add_argument("--backend", default=None, choices=["qwen", "claude"],
                        help="qwen runs locally and costs nothing; claude bills per step.")
    parser.add_argument("--pw-x", default=None)
    parser.add_argument("--bands-x", default=None)
    parser.add_argument("--qe-module", default=None, help="Lmod module to load before running QE.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--structures", type=int, default=51, help="Size of the perturbed ensemble.")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--pretty-print", action="store_true",
                        help="Print a readable run summary on stdout when done; the live step stream goes to stderr.")
    parser.add_argument("--summarize", metavar="PATH", default=None,
                        help="Print the summary of a recorded run (a work dir or log.jsonl) and exit.")
    parser.add_argument("--index", type=int, default=-1,
                        help="With --summarize: which recorded run in the log (default -1, the latest).")
    parser.add_argument("--build-docs", action="store_true",
                        help="Build the documentation index and exit.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.summarize:
        from .report import summarize_log

        try:
            print(summarize_log(args.summarize, index=args.index), end="")
        except (OSError, ValueError, IndexError) as e:
            print(f"chem-llm: cannot summarize {args.summarize}: {e}", file=sys.stderr)
            return 2
        return 0

    settings = Settings.from_yaml(args.config) if args.config else Settings.discover()

    agent = DFTAgent(
        settings,
        model=args.model,
        backend=args.backend,
        pw_x=args.pw_x,
        bands_x=args.bands_x,
        qe_module=args.qe_module,
        max_steps=args.max_steps,
    )
    docs = agent.docs
    if args.build_docs:
        print(docs.build())
        return 0

    if not (args.compound and args.spacegroup):
        build_parser().error("compound and spacegroup are required (or pass --build-docs)")

    if not docs.available:
        print(f"note: no documentation index at {docs.path}; run `chem-llm --build-docs` for better results",
              file=sys.stderr)

    # With --pretty-print the live stream (every step's raw output, tool call
    # and result) goes to stderr, so stdout carries only the final summary.
    stream = contextlib.redirect_stdout(sys.stderr) if args.pretty_print else contextlib.nullcontext()
    with stream:
        result = agent.run(
            args.compound,
            args.spacegroup,
            spacegroup_number=args.spacegroup_number,
            spacegroup_name=args.name,
            work_dir=args.work_dir,
            clear_dir=not args.keep,
            log_file=args.log_file,
            n_structures=args.structures,
            verbose=not args.quiet,
        )
    if args.pretty_print:
        try:
            print(result.report(), end="")
        except Exception as e:  # the run is logged; never leave stdout empty
            print(result)
            print(f"(could not build the run summary: {type(e).__name__}: {e}; "
                  f"retry with: chem-llm --summarize {result.work_dir})")
        return 0 if result.ok else 1
    print(result)
    print("work dir:", result.work_dir)
    if result.plot:
        print("plot    :", result.plot)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
