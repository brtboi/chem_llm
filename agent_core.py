"""Core agent loop: prompt construction, generation, tool-call parsing and
execution, and the step loop itself.

Model loading lives in main.py (the entry point) and is passed in here so
this module has no import-time side effects and can be reused/tested with a
different model or a mock.
"""
import json

from config import MAX_AGENT_STEPS, MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE
from state import AgentState
from tools import TOOLS, TOOL_DISPATCH

SYSTEM_PROMPT_TEMPLATE = (
    "You are a tool-using coding agent that solves tasks step by step.\n"
    "You MUST respond ONLY in valid JSON, one tool call per response.\n"
    "No markdown. No explanations outside the JSON.\n"
    "Output format:\n"
    "{ \"tool\": ..., \"args\": {...} }\n\n"
    "You will be shown the task, your working memory (files touched, notes, "
    "recent tool results), and you must decide the single next tool call.\n\n"
    "RULES YOU MUST FOLLOW:\n"
    "1. Do not call write_file again for a path that already exists in your "
    "working memory unless you are intentionally fixing a bug in it.\n"
    "2. Before calling 'done', you MUST have executed (run_python) every "
    ".py file you wrote, and confirmed its stderr was empty. If a script "
    "produces an output file (e.g. a .cif, .pwi, or .txt file), you must "
    "also read_file that output and visually confirm its contents look "
    "correct for the task -- do not assume success just because stdout/"
    "stderr were empty.\n"
    "3. If run_python returns a non-empty stderr, you must diagnose the "
    "error, fix the underlying script with write_file, and re-run it. Do "
    "not call 'done' while any known error is unresolved. Do not just "
    "write a 'note' about the error and stop -- take a corrective action.\n"
    "4. Do not call 'note' more than once in a row. If you already have a "
    "plan, act on it instead of restating it.\n"
    "5. Only call 'done' when the task is fully complete AND verified: "
    "code ran with no errors, and any output file was read back and checks "
    "out. In your 'done' summary, briefly state what you verified.\n\n"
    f"Available tools:\n{json.dumps(TOOLS, indent=2)}"
)


def build_prompt(state: AgentState, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
        {"role": "user", "content": state.context_summary()},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
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


def run_agent(task: str, model, tokenizer, max_steps: int = MAX_AGENT_STEPS, verbose: bool = True) -> AgentState:
    state = AgentState(task)

    for step in range(1, max_steps + 1):
        prompt = build_prompt(state, tokenizer)
        output = generate(prompt, model, tokenizer)

        if verbose:
            print(f"\n=== STEP {step} ===")
            print("RAW MODEL OUTPUT:\n", output)

        try:
            tool_call = parse_tool_call(output)
        except (json.JSONDecodeError, ValueError) as e:
            # feed the parse failure back in as a note so the model can self-correct
            state.add_note(f"Step {step}: failed to parse model output as JSON: {e}")
            state.log("parse_error", {"raw_output": output[:500]}, str(e))
            continue

        if verbose:
            print("PARSED TOOL CALL:\n", tool_call)

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

    return state