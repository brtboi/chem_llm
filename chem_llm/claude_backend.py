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

# Failures that repeat identically on every later step: billing,
# authentication and permission problems, plus malformed requests.
_TERMINAL_ERRORS = (
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.BadRequestError,
)


class ClaudeModel:
    """Minimal text-in/text-out wrapper around the Messages API.

    Deliberately narrow: the agent protocol lives in the system prompt and
    is parsed out of plain text, exactly as with the local model, so this
    does not use the API's own tool-use or structured-output features.
    """

    # Models whose safety classifiers can decline a request outright. This
    # harness trips `reasoning_extraction` on Opus 5: rule 9 has the agent
    # write its scientific reasoning into `note` steps, and replaying those
    # notes in the context reads to the classifier like an attempt to
    # extract reasoning traces. Measured on one mid-run context: Opus 5
    # refused 2 of 3 attempts, Sonnet 5 and Opus 4.8 zero of 3. Server-side
    # fallbacks let a refused step be answered by another model instead of
    # being lost.
    _REFUSAL_PRONE = ("claude-opus-5", "claude-opus-5-5", "claude-fable-5", "claude-fable-5-1", "claude-mythos-5")

    def __init__(self, model: str = CLAUDE_MODEL, max_tokens: int = CLAUDE_MAX_TOKENS, effort: str = CLAUDE_EFFORT, api_key: str | None = None, use_fallbacks: bool | None = None):
        # api_key=None lets the SDK resolve a credential itself (env var, or
        # an `ant auth login` profile), so an unset ANTHROPIC_API_KEY is not
        # automatically an error.
        self.client = anthropic.Anthropic(api_key=api_key or ANTHROPIC_API_KEY or None)
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.use_fallbacks = model in self._REFUSAL_PRONE if use_fallbacks is None else use_fallbacks
        self.served_by: str | None = None  # model that answered the last call

    def generate(self, system: str, user: str) -> str:
        """Return the concatenated text blocks of one reply.

        No temperature/top_p: sampling parameters are rejected by the
        current Opus/Sonnet models. Thinking is left at the model default
        (adaptive on Opus 5); thinking blocks are skipped here, so only the
        visible text reaches the parser.
        """
        params = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            output_config={"effort": self.effort},
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )

        if self.use_fallbacks:
            # On a policy decline the API re-runs the request on a fallback
            # model within the same call, instead of returning nothing.
            response = self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **params
            )
        else:
            response = self.client.messages.create(**params)

        self.served_by = response.model

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
        except _TERMINAL_ERRORS as e:
            # No credits, bad key, revoked permissions: every later step
            # would fail identically. Stop the loop instead of spending the
            # remaining budget on calls that cannot succeed -- a drained
            # balance once burned 46 of 80 steps this way.
            state.add_note(f"Step {step}: Claude API call failed unrecoverably: {e}")
            state.log("api_error", {"model": claude.model, "fatal": True}, str(e))
            raise StopIteration from e
        except (anthropic.APIStatusError, anthropic.APIConnectionError, RuntimeError) as e:
            # Transient (rate limit, 5xx, dropped connection) or a refusal:
            # record it and let the loop try the next step.
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
