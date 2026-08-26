"""Core agent loop: prompt construction, generation, tool-call parsing and
execution, and the step loop itself.

Model loading lives in main.py (the entry point) and is passed in here so
this module has no import-time side effects and can be reused/tested with a
different model or a mock.

The step loop (`run_step_loop`) is decoupled from *where* each tool call
comes from: `run_agent` below drives it with an LLM-backed source, while
`chem_llm.replay` drives the exact same loop with a source that reads
recorded calls out of a past run's log.jsonl instead. Both share the same
tool execution/logging code path, so a replay is a real re-run of the
recorded tools (not a replay of their recorded results).
"""
import json
import os
from datetime import datetime
from pathlib import Path
import shutil
import time

from . import REPO_ROOT
from .config import MAX_AGENT_STEPS, MAX_HISTORY, MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE, WORK_DIR, MODEL_NAME
from .state import AgentState
from .tools import TOOLS, TOOL_DISPATCH

SYSTEM_PROMPT_TEMPLATE = (
    "You are a tool-using coding agent that solves tasks step by step.\n"
    "Your objective is to complete the task using the fewest necessary tool calls. Once the task is complete and verified, immediately call the \'done\' tool."
    "You MUST respond ONLY in valid JSON, one tool call per response.\n"
    "No markdown. No explanations outside the JSON.\n"
    "Output format:\n"
    "{ \"tool\": ..., \"args\": {...} }\n\n"
    "You will be shown the task, your working memory (files touched, notes, "
    "recent tool results), and you must decide the single next tool call.\n\n"
    "RULES YOU MUST FOLLOW:\n"
    "1. Do not call write_file, make_dir, generate_cif, or get_pseudopotential "
    "again for something that already appears in your working memory's "
    "'Actions already completed' list (a path, or a composition/element you "
    "already fetched) "
    "unless you are intentionally fixing a bug or redoing a failed attempt.\n"
    "2. Before calling 'done', you MUST have executed (run_python) every "
    ".py file you wrote, and confirmed its stderr was empty. If a script "
    "produces an output file (e.g. a .cif, .pwi, or .txt file), you must "
    "also read_file that output and visually confirm its contents look "
    "correct for the task. Do not assume success just because stdout/"
    "stderr were empty.\n"
    "3. If run_python returns a non-empty stderr, do not just tweak the "
    "script and re-run it hoping the error goes away. If the error involves "
    "an API you are not 100% certain about (an unfamiliar function/class/ "
    "method signature, a QE namelist variable, correct units or expected "
    "value ranges), call search_docs on the failing identifier or symptom "
    "first to confirm the correct usage before editing. Then diagnose the "
    "error, write the error and solution in a note, fix the underlying "
    "script with write_file, and re-run it. Do not call 'done' while any "
    "known error is unresolved.\n"
    "4. Once a python runs without errors, note that it has been done successfully "
    "and avoid repeating tasks."
    "5. Do not call 'note' more than once in a row. If you already have a "
    "plan, act on it instead of restating it.\n"
    "6. Do not call the same 'search_docs' query more than once in a row. If you already have the information,"
    "do not call the same search_docs again"
    "7. As soon as the task has been completed and all required verification "
    "has succeeded, your VERY NEXT tool call MUST be 'done'. Do not perform "
    "additional tool calls, extra checks, or exploratory actions after the "
    "task has already been verified.\n"
    "8. Only call 'done' once. The 'done' tool ends the task. In its summary, "
    "briefly state what you accomplished and what you verified. Do NOT call done until you have double checked that EVERY task has been completed\n\n"
    "9. Before every tool call, write in a note outlining any scientific reasoning"
    "needed regarding the tool call parameters."
    f"Available tools:\n{json.dumps(TOOLS, indent=2)}"
)


def build_prompt(state: AgentState, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
        {"role": "user", "content": state.context_summary(max_history = MAX_HISTORY)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )


def parse_tool_call(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end <= start:
            raise
        return json.loads(text[start:end])


def execute_tool(call: dict, state: AgentState):
    tool = call["tool"]
    args = call.get("args", {})

    if tool == "note":
        state.add_note(args.get("text", ""))
        result = "noted"
    elif tool == "done":
        state.done = True
        state.final_result = args.get("summary", "")
        result = state.final_result
    elif tool in TOOL_DISPATCH:
        result = TOOL_DISPATCH[tool](**args)
    else:
        raise ValueError(f"Unknown tool: {tool}")

    state.log(tool, args, result if tool != "note" else "noted")
    return result


def generate(prompt: str, model, tokenizer, remove_prompt_from_output=True, print_generated_tokens=False):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=DO_SAMPLE,
    )
    if print_generated_tokens:
        print(f"Generated tokens: {outputs[0].shape[0] - input_len}")
    if remove_prompt_from_output:
        decoded = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    else:
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.strip()


def make_llm_step_source(model, tokenizer, verbose: bool = True):
    """Build a `next_tool_call(state, step)` source for `run_step_loop` that
    asks the model to generate each step, same as the original inline
    run_agent loop. A JSON parse failure is logged as a 'parse_error' step
    (fed back as a note so the model can self-correct) and reported to the
    loop as a no-op step rather than a tool call.
    """
    def next_tool_call(state: AgentState, step: int):
        prompt = build_prompt(state, tokenizer)
        output = generate(prompt, model, tokenizer)
        if verbose:
            print("RAW MODEL OUTPUT:\n", output)
        try:
            return parse_tool_call(output)
        except (json.JSONDecodeError, ValueError) as e:
            state.add_note(f"Step {step}: failed to parse model output as JSON: {e}")
            state.log("parse_error", {"raw_output": output[:500]}, str(e))
            return None

    return next_tool_call


def run_step_loop(state: AgentState, next_tool_call, max_steps: int, verbose: bool = True) -> None:
    """Drive `state` through up to `max_steps` tool calls, one per step,
    obtained from `next_tool_call(state, step)`. Mutates `state` in place.

    `next_tool_call` should return a `{"tool": ..., "args": ...}` dict for
    the step, `None` if it already handled the step itself (e.g. logged a
    parse failure as a note) and no tool should be executed, or raise
    `StopIteration` to end the loop early (a replay source runs out of
    recorded calls before `max_steps`).
    """
    for step in range(1, max_steps + 1):
        if verbose:
            print(f"\n=== STEP {step} ===")

        try:
            tool_call = next_tool_call(state, step)
        except StopIteration:
            break

        if tool_call is None:
            continue

        if verbose:
            print("TOOL CALL:\n", tool_call)

        try:
            result = execute_tool(tool_call, state)
        except Exception as e:
            result = f"ERROR executing {tool_call.get('tool')}: {e}"
            state.log(tool_call.get("tool", "unknown"), tool_call.get("args", {}), result)

        if verbose:
            print("TOOL RESULT:\n", result)

        if state.done:
            break
    else:
        state.add_note("Max steps reached without explicit 'done' call.")


# Directories clear_dir refuses to touch: the target itself, or the target
# being an ancestor of any of these (which would mean clearing it wipes out
# the filesystem root, the user's home directory, or the whole repo).
_DANGEROUS_CLEAR_ANCESTORS = (Path("/"), Path.home().resolve(), REPO_ROOT)


def _guard_clear_target(directory: Path) -> None:
    directory = directory.resolve()
    if any(directory == root or root.is_relative_to(directory) for root in _DANGEROUS_CLEAR_ANCESTORS):
        raise RuntimeError(
            f"Refusing clear_dir on {directory}: it is, or contains, the "
            "filesystem root / your home directory / the repo root. "
            "clear_dir is meant for an agent scratch directory (e.g. "
            "sandbox/testN), not this."
        )


def _confirm_clear_dir(directory: Path) -> bool:
    response = input(
        f"clear_dir=True: about to permanently delete everything in "
        f"{directory} except *.jsonl log files. Type 'yes' to continue: "
    )
    return response.strip().lower() == "yes"


def clear_directory(directory: Path) -> None:
    """Delete every top-level entry in `directory` except *.jsonl log
    files, after an interactive stdin confirmation. Raises RuntimeError if
    the user declines, or if `directory` fails the dangerous-target guard.
    """
    _guard_clear_target(directory)

    if not _confirm_clear_dir(directory):
        raise RuntimeError(f"clear_dir cancelled by user for {directory}")

    for entry in directory.iterdir():
        if entry.is_file() and entry.suffix == ".jsonl":
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()

def copy_to_cwd(load_path) -> None:
    if load_path is None:
        return

    src = Path(load_path)
    dst = Path.cwd() / src.name

    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)


def prepare_work_dir(work_dir: Path, clear_dir: bool = False, load_path=None) -> None:
    """Ensure `work_dir` exists and make it the current directory -- every
    tool (write_file, run_python, generate_cif, ...) resolves its paths
    against cwd, so this is the one place that contract is established,
    rather than every entry point (run_agent, replay) chdir'ing on its own.
    Optionally clears it first and/or seeds it from `load_path`.
    """
    if not work_dir.exists():
        print(f"Work directory {work_dir} does not exist yet -- creating it.")
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)

    if clear_dir:
        clear_directory(Path.cwd())

    copy_to_cwd(load_path)


def build_log_entry(state: AgentState, start_time: float, work_dir: Path, clear_dir: bool, load_path, model_name: str, extra: dict | None = None) -> dict:
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "work_dir": str(work_dir),
        "clear_dir": clear_dir,
        "load_path": bool(load_path),
        "runtime": time.perf_counter() - start_time,
        "num_steps": len(state.history),
        "completed": state.done,
        "final_state": state.to_dict(),
    }
    if extra:
        entry.update(extra)
    return entry


def append_log(log_file: Path, entry: dict) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        json.dump(entry, f)
        f.write("\n")


_UNSET = object()


def run_agent(
    task: str,
    model,
    tokenizer,
    max_steps: int = MAX_AGENT_STEPS,
    verbose: bool = True,
    work_dir: Path = WORK_DIR,
    log_file=_UNSET,
    clear_dir: bool = False,
    load_path=None,
) -> AgentState:
    """Run the agent live: the model generates each tool call. Executes in
    `work_dir` (defaults to config.WORK_DIR) and appends a run record to
    `log_file` (defaults to `work_dir / "log.jsonl"`; pass `log_file=None`
    to skip logging).
    """
    if log_file is _UNSET:
        log_file = work_dir / "log.jsonl"

    prepare_work_dir(work_dir, clear_dir, load_path)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.touch(exist_ok=True)

    start_time = time.perf_counter()
    state = AgentState(task)

    run_step_loop(state, make_llm_step_source(model, tokenizer, verbose), max_steps, verbose)

    if log_file:
        print("logging to...", log_file)
        append_log(log_file, build_log_entry(state, start_time, work_dir, clear_dir, load_path, MODEL_NAME))

    return state
