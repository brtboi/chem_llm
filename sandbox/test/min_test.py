#!/usr/bin/env python
# coding: utf-8

# In[14]:


# Set LLM download destination
import os
os.environ["HF_HOME"] = "/pscratch/sd/b/brenthu/huggingface"
os.chdir("test_si")
print(os.getcwd())


# In[2]:


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import subprocess
import os
import sys
import datetime


# In[3]:


HF_TOKEN = "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"
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    token=HF_TOKEN
)


# In[4]:


# MODEL SETTINGS
MAX_NEW_TOKENS = 5000
TEMPERATURE = 0.0
DO_SAMPLE = False
MAX_AGENT_STEPS = 16

# TOOL SETTINGS
READ_MAX_CHARS = 10000
# TODO: file prefix for writing files?


# In[5]:


class AgentState:
    """External working memory for the agent loop.

    Tracks the running task, every tool call + result, a file registry
    (so the model doesn't have to guess what exists), and a freeform
    scratchpad the model can write to via the `note` tool.
    """
    def __init__(self, task: str):
        self.task = task
        self.created_at = datetime.datetime.now().isoformat()
        self.history: list[dict] = []           # full tool call/result log
        self.files: dict[str, str] = {}         # path -> last known status
        self.last_errors: dict[str, str] = {}   # path -> last stderr/execption
        self.scratchpad: list[str] = []         # freeform notes from the model
        self.consecutive_notes = 0
        self.done = False
        self.final_result = None

    def log(self, tool: str, args: dict, result):
        entry = {"step": len(self.history) + 1, "tool": tool, "args": args, "result": result}
        self.history.append(entry)
        if tool == "note":
            self.consecutive_notes += 1
        else:
            self.consecutive_notes = 0
        if tool == "write_file":
            self.files[args.get("path", "?")] = "written"
        if tool == "run_python" and isinstance(result, dict) and result.get("stderr"):
            self.last_errors[args.get("path", "?")] = result["stderr"]
        elif tool == "run_python":
            self.last_errors.pop(args.get("path", "?"), None)

    def add_note(self, text: str):
        self.scratchpad.append(text)

    def context_summary(self, max_history: int = 6) -> str:
        """Compact summary fed back into the prompt each turn."""
        recent = self.history[-max_history:]
        lines = [f"Task: {self.task}"]
        if self.files:
            lines.append(f"Known files: {json.dumps(self.files)}")
            lines.append(
                "Do NOT call write_file again for a path that already appears above "
                "unless you are intentionally fixing a bug in it. If a file already "
                "exists and you need to check it, use read_file instead of rewriting it."
            )
        if self.last_errors:
            lines.append(f"Unresolved errors: {json.dumps(self.last_errors)}")
        if self.scratchpad:
            lines.append("Notes so far:")
            lines.extend(f"  - {n}" for n in self.scratchpad)
        if recent:
            lines.append("Recent tool calls:")
            for entry in recent:
                lines.append(
                    f"  step {entry['step']}: {entry['tool']}({entry['args']}) "
                    f"-> {str(entry['result'])}"
                )
        else:
            lines.append("No tool calls yet.")

        if self.consecutive_notes >= 1:
            lines.append(
                "WARNING: you have already called 'note' without taking action. "
                "Do not call 'note' again next unless you have a good reason to do so — call write_file, run_python, read_file, or done."
            )
        return "\n".join(lines)

    def to_dict(self):
        return {
            "task": self.task,
            "created_at": self.created_at,
            "history": self.history,
            "files": self.files,
            "scratchpad": self.scratchpad,
            "done": self.done,
            "final_result": self.final_result,
        }


# In[6]:


def write_file(path: str, content: str):
    with open(path, "w") as f:
        f.write(content)
    return f"Wrote {path}"

def read_file(path: str, max_chars: int = READ_MAX_CHARS):
    if not os.path.exists(path):
        return f"ERROR: {path} does not exist"
    with open(path, "r", errors="replace") as f:
        content = f.read()
    truncated = len(content) > max_chars
    return {
        "path": path,
        "content": content[:max_chars],
        "truncated": truncated,
        "total_chars": len(content),
    }

def run_python(path: str):
    result = subprocess.run(
        [sys.executable, path],
        capture_output=True,
        text=True
    )
    return {"stdout": result.stdout, "stderr": result.stderr}


# In[7]:


TOOLS = [
    {
        "name": "write_file",
        "description": "Write a file to disk",
        "parameters": {"path": "string", "content": "string"}
    },
    {
        "name": "read_file",
        "description": "Read a file from disk (truncated if very large)",
        "parameters": {"path": "string"}
    },
    {
        "name": "run_python",
        "description": "Execute a python file, returns stdout/stderr",
        "parameters": {"path": "string"}
    },
    {
        "name": "note",
        "description": "Record a observation/plan in your scratchpad without taking an action",
        "parameters": {"text": "string"}
    },
    {
        "name": "done",
        "description": "Call this when the task is fully complete. Provide a summary.",
        "parameters": {"summary": "string"}
    }
]

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


# In[8]:


def build_prompt(state: AgentState):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
        {"role": "user", "content": state.context_summary()},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt


# In[9]:


def parse_tool_call(text: str):
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

    if tool == "write_file":
        result = write_file(**args)
    elif tool == "read_file":
        result = read_file(**args)
    elif tool == "run_python":
        result = run_python(**args)
    elif tool == "note":
        state.add_note(args.get("text", ""))
        result = "noted"
    elif tool == "done":
        state.done = True
        state.final_result = args.get("summary", "")
        result = state.final_result
    else:
        raise ValueError(f"Unknown tool: {tool}")

    if tool != "note":  # notes are logged via add_note already reflected in state
        state.log(tool, args, result)
    else:
        state.log(tool, args, "noted")
    return result


# In[10]:


def generate(prompt: str, remove_prompt_from_output=True, printGeneratedTokens=False):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=DO_SAMPLE
    )
    if printGeneratedTokens:
        print(f"Generated tokens: {outputs[0].shape[0] - input_len}")
    if remove_prompt_from_output:
        decoded = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    else:
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.strip()


# In[11]:


def run_agent(task: str, max_steps: int = MAX_AGENT_STEPS, verbose: bool = True):
    state = AgentState(task)

    for step in range(1, max_steps + 1):
        prompt = build_prompt(state)
        # output = generate(prompt, printGeneratedTokens=True)
        output = generate(prompt)

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


# In[15]:


final_state = run_agent("""
    Your task is to create a version of the existing workflow in the ../example/ directory for a simple silicon crystal while preserving overall functionality
    Please open and creafully read ../example/generate_structures.py and ../example/setup_jobs.py.
    Directories ./calculations and ./template has already been made for you.
    Please use the following pseudopotential file: ./template/Si.upf.
    Write in a note the function of each constant, function, randomization, including whether it is specific for CsPbBr3 or generic,
    determine which parts must be rewritten for the new target compound listed above.
    Then, write to a new file a python script that follows the same workflow structure but for the new target compound.
    Please review to make sure it works and then run generate_structures.py, setup_jobs.py
    In the final step, please provide a concise summary of the changes made, any assumptions made, and any remaining uncertainties requiring domain expertise
""", verbose=False)
print("\nFINAL STATE:\n", json.dumps(final_state.to_dict(), indent=2))

