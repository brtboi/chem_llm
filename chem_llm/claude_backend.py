"""Claude API backend for the agent loop -- a drop-in alternative to the
local Qwen model in agent_core.

Everything downstream of text generation is unchanged: the same
SYSTEM_PROMPT_TEMPLATE, the same AgentState.context_summary() as the user
turn, the same parse_tool_call, the same tools, the same run_step_loop and
log.jsonl records. Only the "prompt in, text out" step differs -- instead of
a chat template plus HF generate(), the prompt goes to the Messages API as
(system, user) and the text blocks of the reply come back.

    from chem_llm.claude_backend import ClaudeModel, make_claude_step_source
    from chem_llm.agent_core import run_agent

    claude = ClaudeModel()
    run_agent(TASK, step_source=make_claude_step_source(claude),
              model_name=claude.model, clear_dir=True)

Requires an Anthropic credential: ANTHROPIC_API_KEY (via .env, which
chem_llm.config already loads) or any other source the SDK resolves.
"""
import json

import anthropic

from .agent_core import SYSTEM_PROMPT_TEMPLATE, parse_tool_call
from .config import ANTHROPIC_API_KEY, CLAUDE_EFFORT, CLAUDE_MAX_TOKENS, CLAUDE_MODEL, MAX_HISTORY
from .state import AgentState


class ClaudeModel:
    """Minimal text-in/text-out wrapper around the Messages API.

    Deliberately narrow: the agent protocol lives in the system prompt and
    is parsed out of plain text, exactly as with the local model, so this
    does not use the API's own tool-use or structured-output features.
    """

    def __init__(self, model: str = CLAUDE_MODEL, max_tokens: int = CLAUDE_MAX_TOKENS, effort: str = CLAUDE_EFFORT, api_key: str | None = None):
        # api_key=None lets the SDK resolve a credential itself (env var, or
        # an `ant auth login` profile), so an unset ANTHROPIC_API_KEY is not
        # automatically an error.
        self.client = anthropic.Anthropic(api_key=api_key or ANTHROPIC_API_KEY or None)
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort

    def generate(self, system: str, user: str) -> str:
        """Return the concatenated text blocks of one reply.

        No temperature/top_p: sampling parameters are rejected by the
        current Opus/Sonnet models. Thinking is left at the model default
        (adaptive on Opus 5); thinking blocks are skipped here, so only the
        visible text reaches the parser.
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            output_config={"effort": self.effort},
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise RuntimeError(f"Claude refused this step: {getattr(details, 'category', None)} {getattr(details, 'explanation', '')}")

        return "".join(block.text for block in response.content if block.type == "text").strip()


def make_claude_step_source(claude: ClaudeModel, verbose: bool = True, max_history: int = MAX_HISTORY):
    """Build a `next_tool_call(state, step)` source for `run_step_loop`,
    the Claude-backed counterpart of agent_core.make_llm_step_source.

    Mirrors it exactly, including how a malformed reply is handled: logged
    as a 'parse_error' step and fed back as a note so the model can
    self-correct, with no tool executed for that step.
    """
    def next_tool_call(state: AgentState, step: int):
        try:
            output = claude.generate(SYSTEM_PROMPT_TEMPLATE, state.context_summary(max_history=max_history))
        except (anthropic.APIStatusError, anthropic.APIConnectionError, RuntimeError) as e:
            # Treat an API failure like an unparseable step rather than
            # killing the run: the loop records it and tries the next step.
            state.add_note(f"Step {step}: Claude API call failed: {e}")
            state.log("api_error", {"model": claude.model}, str(e))
            return None

        if verbose:
            print("RAW MODEL OUTPUT:\n", output)

        try:
            return parse_tool_call(output)
        except (json.JSONDecodeError, ValueError) as e:
            state.add_note(f"Step {step}: failed to parse model output as JSON: {e}")
            state.log("parse_error", {"raw_output": output[:500]}, str(e))
            return None

    return next_tool_call
