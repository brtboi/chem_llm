"""The LangChain wrappers around each backend, without loading real weights."""
import pytest
import torch
from langchain_core.messages import HumanMessage, SystemMessage

from chem_llm.agent_core import build_prompt, make_chat_step_source
from chem_llm.llm import ClaudeChatModel, TransformersChatModel, check_server, vllm_chat_model
from chem_llm.state import AgentState


class Enc(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert enable_thinking is False and add_generation_prompt
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)

    def __call__(self, text, return_tensors):
        return Enc(input_ids=torch.zeros((1, 7), dtype=torch.long))

    def decode(self, ids, skip_special_tokens):
        return ' {"tool": "done", "args": {"summary": "x"}} '


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return torch.zeros((1, 10), dtype=torch.long)


def test_transformers_model_matches_the_original_prompt_and_counts_tokens():
    tok, model = FakeTokenizer(), FakeModel()
    llm = TransformersChatModel(hf_model=model, tokenizer=tok, max_new_tokens=99, temperature=0.3)
    state = AgentState("task text")
    from chem_llm.agent_core import build_messages

    assert llm.prompt_text(build_messages(state, 32, True)) == build_prompt(state, tok, 32, True)
    reply = llm.invoke([SystemMessage("s"), HumanMessage("u")])
    assert reply.content == '{"tool": "done", "args": {"summary": "x"}}'
    assert reply.usage_metadata == {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
    assert model.kwargs["max_new_tokens"] == 99 and model.kwargs["temperature"] == 0.3


def test_greedy_decoding_does_not_pass_temperature():
    model = FakeModel()
    TransformersChatModel(hf_model=model, tokenizer=FakeTokenizer(), do_sample=False).invoke("hi")
    assert model.kwargs["do_sample"] is False and "temperature" not in model.kwargs


class StubClaude:
    model = "claude-stub"

    def __init__(self, fail=None):
        self.fail, self.calls = fail, []
        self.last_usage = {"input_tokens": 100, "output_tokens": 5, "total_tokens": 105}

    def generate(self, system, user):
        self.calls.append((system, user))
        if self.fail:
            raise self.fail
        return '{"tool": "done", "args": {"summary": "x"}}'


def test_claude_wrapper_sends_system_and_user_separately():
    claude = StubClaude()
    reply = ClaudeChatModel(claude=claude).invoke([SystemMessage("SYS"), HumanMessage("USER")])
    assert claude.calls == [("SYS", "USER")]
    assert reply.usage_metadata["input_tokens"] == 100


def test_claude_refusal_is_logged_and_skipped():
    pytest.importorskip("anthropic")
    from chem_llm.claude_backend import make_claude_step_source

    state = AgentState("t")
    source = make_claude_step_source(StubClaude(fail=RuntimeError("Claude refused this step")), verbose=False)
    assert source(state, 1) is None
    assert state.history[-1]["tool"] == "api_error"


def test_chat_step_source_records_usage():
    from chem_llm.testing import ScriptedChatModel

    state = AgentState("t")
    source = make_chat_step_source(ScriptedChatModel(replies=[{"tool": "done", "args": {}}]), verbose=False)
    assert source(state, 1) == {"tool": "done", "args": {}}
    assert state.llm_usage["prompt_tokens"] > 1000 and state.llm_usage["prompt_chars"] > 4000


def test_vllm_client_turns_thinking_off():
    llm = vllm_chat_model("http://localhost:1/v1", "Qwen/X", max_tokens=5000, temperature=0.3, do_sample=True)
    assert llm.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert llm.max_tokens == 5000 and llm.temperature == 0.3
    greedy = vllm_chat_model("http://localhost:1/v1", "Qwen/X", max_tokens=5, temperature=0.3, do_sample=False)
    assert greedy.temperature == 0.0


def test_missing_vllm_server_explains_how_to_start_one():
    with pytest.raises(ConnectionError, match="serve_vllm.sh"):
        check_server("http://127.0.0.1:9/v1", timeout=1)


def test_vllm_backend_refuses_a_server_with_another_model(settings, monkeypatch):
    import chem_llm.llm as llm_module
    from chem_llm import DFTAgent

    monkeypatch.setattr(llm_module, "check_server", lambda url: ["some/other-model"])
    with pytest.raises(ValueError, match="serves"):
        DFTAgent(settings, backend="vllm").load()
