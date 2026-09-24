"""Reproduce a past agent run by re-executing its recorded tool calls, in
order, without asking an LLM to generate them.

Every run_agent() call appends one line to a log.jsonl -- a JSON object
whose `final_state.history` is the ordered list of `{step, tool, args,
result}` calls that run made. `replay_run` reads that list out of a chosen
run and drives it back through `run_step_loop` (the same loop `run_agent`
uses), pulling `{tool, args}` from the recording instead of the model. Tools
are executed for real, so this reissues the same generate_cif/run_python/
write_file/... calls with the same arguments -- it does not merely print
back the old results.

CLI usage:
    python -m chem_llm.replay sandbox/test17                # replay latest run in that dir's log.jsonl
    python -m chem_llm.replay sandbox/test17 --index 0       # replay the first recorded run
    python -m chem_llm.replay sandbox/test17/log.jsonl -i -2 # a log.jsonl path works too
    python -m chem_llm.replay sandbox/test17 --use-config-work-dir  # execute in config.WORK_DIR instead
"""
import argparse
import json
import time
from pathlib import Path

from . import REPO_ROOT, config
from .agent_core import append_log, build_log_entry, prepare_work_dir, run_step_loop
from .state import AgentState


def resolve_log_file(path: Path) -> Path:
    """Accept either a log.jsonl path directly or a directory containing
    one (e.g. a sandbox/testN directory)."""
    path = Path(path)
    if path.is_dir():
        path = path / "log.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No log file found at {path}")
    return path


def load_run(log_file: Path, index: int) -> dict:
    """Load a single recorded run (one jsonl line) from `log_file`. `index`
    supports Python-style negative indexing (-1 = most recent run)."""
    with log_file.open(encoding="utf-8") as f:
        runs = [json.loads(line) for line in f if line.strip()]

    if not runs:
        raise ValueError(f"{log_file} has no recorded runs.")

    try:
        return runs[index]
    except IndexError:
        raise IndexError(
            f"{log_file} has {len(runs)} recorded run(s); index {index} is out of range."
        ) from None


def make_replay_step_source(recorded_history: list[dict], verbose: bool = True):
    """Build a `next_tool_call(state, step)` source for `run_step_loop` that
    replays `recorded_history` (an `AgentState.history`-shaped list, pulled
    from a past run's `final_state.history`) instead of generating calls."""

    def next_tool_call(state: AgentState, step: int):
        if step > len(recorded_history):
            raise StopIteration

        entry = recorded_history[step - 1]
        tool, args, result = entry["tool"], entry.get("args", {}), entry.get("result")

        if verbose:
            print(f"REPLAYING recorded step {entry.get('step', step)}: {tool}({args})")

        if tool == "parse_error":
            # Not a real tool call: the original run failed to parse a model
            # response at this step. Reproduce the note/log entry rather
            # than trying to "execute" it.
            state.add_note(f"Step {step}: failed to parse model output as JSON (replayed): {result}")
            state.log("parse_error", args, result)
            return None

        return {"tool": tool, "args": args}

    return next_tool_call


def replay_run(
    entry: dict,
    work_dir: Path,
    verbose: bool = True,
    clear_dir: bool = False,
    load_path=None,
    source_log_file: Path | None = None,
    source_index: int | None = None,
) -> AgentState:
    """Re-execute one recorded run's tool calls, in `work_dir`, and append
    the result to `work_dir / "log.jsonl"`."""
    final_state = entry.get("final_state", {})
    task = final_state.get("task", "")
    recorded_history = final_state.get("history", [])

    prepare_work_dir(work_dir, clear_dir, load_path)
    log_file = work_dir / "log.jsonl"

    start_time = time.perf_counter()
    state = AgentState(task)
    step_source = make_replay_step_source(recorded_history, verbose=verbose)
    run_step_loop(state, step_source, max_steps=len(recorded_history), verbose=verbose)

    log_entry = build_log_entry(
        state,
        start_time,
        work_dir,
        clear_dir,
        load_path,
        model_name=entry.get("model", config.MODEL_NAME),
        extra={
            "replayed_from": {
                "log_file": str(source_log_file) if source_log_file else None,
                "run_index": source_index,
                "original_timestamp": entry.get("timestamp"),
            }
        },
    )
    print("logging to...", log_file)
    append_log(log_file, log_entry)

    return state


def sandbox_dir(n: int) -> Path:
    """Resolve an integer to its sandbox/testN directory, anchored to
    REPO_ROOT so this works regardless of cwd (e.g. 17 -> sandbox/test17)."""
    return REPO_ROOT / "sandbox" / f"test{n}"


def copy_parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a recorded run from one sandbox/testN directory into "
            "another, creating the destination directory if it doesn't exist."
        ),
    )
    parser.add_argument(
        "from_dir", type=int,
        help="Source sandbox test number to replay from, e.g. 17 for sandbox/test17.",
    )
    parser.add_argument(
        "to_dir", type=int,
        help="Destination sandbox test number to replay into, e.g. 25 for "
             "sandbox/test25. Created if it doesn't already exist.",
    )
    parser.add_argument(
        "-i", "--index",
        type=int,
        default=-1,
        help="Which recorded run in the source log.jsonl to replay: a 0-based line "
             "number, negative counts from the end. Default: -1 (most recent).",
    )
    parser.add_argument(
        "--clear-dir",
        action="store_true",
        help="Clear the destination directory before replaying. Off by default.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step console output.",
    )
    return parser.parse_args(argv)


def copy_main(argv=None) -> None:
    args = copy_parse_args(argv)

    source_dir = sandbox_dir(args.from_dir)
    dest_dir = sandbox_dir(args.to_dir)
    log_file = resolve_log_file(source_dir)
    entry = load_run(log_file, args.index)

    print(f"Replaying run {args.index} from {log_file} into {dest_dir}")

    state = replay_run(
        entry,
        work_dir=dest_dir,
        verbose=not args.quiet,
        clear_dir=args.clear_dir,
        source_log_file=log_file,
        source_index=args.index,
    )

    print("\nFINAL STATE:\n", json.dumps(state.to_dict(), indent=2))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a past agent run by re-executing its recorded tool "
            "calls in order, without invoking the LLM."
        ),
    )
    parser.add_argument(
        "log_path",
        type=Path,
        help="Path to a log.jsonl file, or a directory containing one (e.g. sandbox/test17).",
    )
    parser.add_argument(
        "-i", "--index",
        type=int,
        default=-1,
        help="Which recorded run to replay: a 0-based line number in the jsonl file, "
             "negative counts from the end. Default: -1 (the most recent run).",
    )
    parser.add_argument(
        "--use-config-work-dir",
        action="store_true",
        help="Execute the replay in chem_llm.config.WORK_DIR instead of the directory "
             "that owns log_path (the default: replay in place).",
    )
    parser.add_argument(
        "--clear-dir",
        action="store_true",
        help="Clear the execution directory before replaying (same destructive, "
             "confirmation-gated behavior as a live run_agent(clear_dir=True) call). "
             "Off by default.",
    )
    parser.add_argument(
        "--load-path",
        type=Path,
        default=None,
        help="Copy this file/directory into the execution directory before replaying "
             "(mirrors run_agent's load_path). Off by default.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step console output.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    log_file = resolve_log_file(args.log_path)
    entry = load_run(log_file, args.index)
    work_dir = config.WORK_DIR if args.use_config_work_dir else log_file.parent.resolve()

    print(f"Replaying run {args.index} from {log_file} into {work_dir}")

    state = replay_run(
        entry,
        work_dir=work_dir,
        verbose=not args.quiet,
        clear_dir=args.clear_dir,
        load_path=args.load_path,
        source_log_file=log_file,
        source_index=args.index,
    )

    print("\nFINAL STATE:\n", json.dumps(state.to_dict(), indent=2))


if __name__ == "__main__":
    main()
