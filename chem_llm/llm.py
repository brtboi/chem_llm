"""The agent's language models, as LangChain chat models.

Every backend is a langchain_core BaseChatModel, so the agent loop has one
interface ("messages in, AIMessage out, with usage_metadata") whatever
generates the text, and LangChain's tracing (LangSmith, Langfuse callbacks)
sees every call:

    qwen     TransformersChatModel -- the weights loaded in this process
             with transformers. Simple, but slow: no batching, no prompt
             caching, ~50 s per agent step for the 27B model.
    vllm     langchain_openai.ChatOpenAI against a vLLM server
             (scripts/serve_vllm.sh). Same weights, much faster: paged
             attention, and the system prompt, identical on every step, is
             computed once (prefix caching).
    claude   chem_llm.claude_backend (the Anthropic API; bills per step).

Any other LangChain chat model can be passed to DFTAgent(llm=...) as well,
e.g. a langchain_core fake model in tests (see chem_llm.testing).
"""
from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict

_ROLES = {"system": "system", "human": "user", "ai": "assistant"}


def to_chat_dicts(messages: list[BaseMessage]) -> list[dict]:
    """LangChain messages -> the [{"role", "content"}] list chat templates take."""
    return [{"role": _ROLES.get(m.type, m.type), "content": m.content} for m in messages]


class TransformersChatModel(BaseChatModel):
    """A transformers causal LM loaded in this process, as a chat model.

    Produces exactly the prompt the original loop built: the tokenizer's
    chat template with Qwen's thinking mode off.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hf_model: Any
    tokenizer: Any
    model_name: str = "local"
    max_new_tokens: int = 5000
    temperature: float = 0.3
    do_sample: bool = True

    @property
    def _llm_type(self) -> str:
        return "transformers"

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name, "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature, "do_sample": self.do_sample}

    def prompt_text(self, messages: list[BaseMessage]) -> str:
        return self.tokenizer.apply_chat_template(
            to_chat_dicts(messages), tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self.prompt_text(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.hf_model.device)
        n_in = inputs["input_ids"].shape[1]
        sampling = {"do_sample": self.do_sample}
        if self.do_sample:
            # Greedy decoding ignores temperature, and transformers warns
            # about it on every step if it is passed anyway.
            sampling["temperature"] = self.temperature
        out = self.hf_model.generate(**inputs, max_new_tokens=self.max_new_tokens, **sampling)
        new = out[0][n_in:]
        text = self.tokenizer.decode(new, skip_special_tokens=True).strip()
        usage = {"input_tokens": int(n_in), "output_tokens": int(new.shape[0]),
                 "total_tokens": int(n_in + new.shape[0])}
        message = AIMessage(content=text, usage_metadata=usage)
        return ChatResult(generations=[ChatGeneration(message=message)])


class ClaudeChatModel(BaseChatModel):
    """chem_llm.claude_backend.ClaudeModel as a chat model: the system
    message becomes the API's system prompt, the rest the user turn.
    Errors (billing, refusals, rate limits) propagate unchanged; the
    Claude step source in claude_backend decides which ones end a run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    claude: Any

    @property
    def _llm_type(self) -> str:
        return "anthropic-messages"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        system = "\n\n".join(m.content for m in messages if m.type == "system")
        user = "\n\n".join(m.content for m in messages if m.type != "system")
        text = self.claude.generate(system, user)
        usage = getattr(self.claude, "last_usage", None)
        message = AIMessage(content=text, usage_metadata=usage) if usage else AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=message)])


def vllm_chat_model(base_url: str, model: str, *, max_tokens: int, temperature: float, do_sample: bool,
                    timeout: float = 1800):
    """A ChatOpenAI client for a vLLM server, with the same decoding as the
    local transformers backend (Qwen thinking off; greedy if not sampling)."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=base_url,
        api_key="EMPTY",            # vLLM does not check it unless started with --api-key
        model=model,
        max_tokens=max_tokens,
        temperature=temperature if do_sample else 0.0,
        timeout=timeout,
        max_retries=2,
        # vLLM forwards chat_template_kwargs to the chat template, as
        # apply_chat_template(..., enable_thinking=False) does locally.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def check_server(base_url: str, timeout: float = 5.0) -> list[str]:
    """Model ids a running OpenAI-compatible server offers; raises
    ConnectionError with instructions if nothing answers."""
    import json
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return [m["id"] for m in json.load(r).get("data", [])]
    except (urllib.error.URLError, OSError) as e:
        raise ConnectionError(
            f"no vLLM server answering at {base_url} ({e}). Start one on this node with "
            "scripts/serve_vllm.sh (it takes a few minutes to load the weights), or set "
            "model.backend: qwen to load the model in-process instead."
        ) from e
