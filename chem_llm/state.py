import datetime
import json

from .tools import REMEMBER_FIELDS

class AgentState:
    def __init__(self, task: str):
        self.task = task
        self.created_at = datetime.datetime.now().isoformat()
        self.history: list[dict] = []           # full tool call/result log
        self.files: dict[str, str] = {}         # path -> last known status
        self.last_errors: dict[str, str] = {}   # path -> last stderr/exception
        self.completed_actions: dict[str, str] = {}  # memory key -> summary, survives history truncation
        self.scratchpad: list[str] = []         # freeform notes from the model
        self.consecutive_notes = 0
        self.done = False
        self.final_result = None

    @staticmethod
    def _memory_key(tool: str, args: dict) -> str | None:
        """Build a stable identity key for a tool call from the arg names it
        registered via `register_tool(..., remember_on=...)`, or None if the
        tool didn't opt in to being remembered."""
        fields = REMEMBER_FIELDS.get(tool)
        if fields is None:
            return None
        return f"{tool}:" + ":".join(str(args.get(f)) for f in fields)

    @staticmethod
    def _summarize_result(result) -> str:
        if isinstance(result, dict):
            keys = ("output_path", "path", "selected_material_id", "space_group", "hints")
            summary = {k: result[k] for k in keys if k in result}
            return json.dumps(summary) if summary else str(result)[:200]
        return str(result)[:200]

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

        key = self._memory_key(tool, args)
        if key is not None:
            failed = (isinstance(result, dict) and result.get("success") is False) or (
                isinstance(result, str) and result.startswith("ERROR")
            )
            if failed:
                # don't remember failures as "done" -- let the model retry
                self.completed_actions.pop(key, None)
            else:
                self.completed_actions[key] = (
                    f"{tool}({args}) succeeded -> {self._summarize_result(result)}"
                )

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
        if self.completed_actions:
            lines.append(
                "Actions already completed successfully this task, including where "
                "their output was saved (do NOT repeat any of these -- e.g. do not "
                "call generate_cif/get_pseudopotential again for the same "
                "composition/element, or make_dir again for the same path -- unless "
                "you are intentionally redoing it to fix a mistake):"
            )
            lines.extend(f"  - {v}" for v in self.completed_actions.values())
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