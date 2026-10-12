"""The agent loop as a LangGraph state graph.

    START -> think -> act -> think -> ... -> END

think   asks the step source for the next tool call (the model, or a
        replayed recording). A reply that is not a tool call was already
        logged by the source; the loop goes straight back to think.
act     runs the tool through agent_core.execute_tool (harness refusals
        included). The run ends when the agent calls done, the step budget
        is spent, or a replay source runs out.

The graph carries only control flow (step number, pending call, why it
stopped). The agent's memory -- history, notes, files -- is the AgentState
the tools and prompt builder already use, held by the nodes. Each node is
timed by a telemetry.StepRecorder.

Tracing: with LANGSMITH_TRACING=true and LANGSMITH_API_KEY set, LangChain
sends each run (graph, nodes and every model call, with prompts) to
LangSmith. Off unless those are set.
"""
from __future__ import annotations

import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class LoopState(TypedDict, total=False):
    step: int
    call: dict | None
    stop: str | None   # "done" | "max_steps" | "source_exhausted"


def build_graph(state, next_tool_call, *, max_steps: int, verbose: bool = True, observer=None, recorder=None):
    """Compile the loop for one run. `state` is the AgentState the run
    mutates; see agent_core.run_step_loop for the other arguments."""
    from .agent_core import execute_tool

    def report(step: int, before: int) -> None:
        if observer is not None:
            for entry in state.history[before:]:
                observer.finish(step, entry)

    def end_step(step: int, before: int) -> None:
        report(step, before)
        if recorder is not None:
            recorder.end()

    def think(s: LoopState) -> LoopState:
        step = s.get("step", 0) + 1
        if verbose:
            print(f"\n=== STEP {step} ===")
        before = len(state.history)
        if recorder is not None:
            recorder.begin(step)
        t = time.perf_counter()
        try:
            call = next_tool_call(state, step)
        except StopIteration:
            if recorder is not None:
                recorder.llm_done(time.perf_counter() - t)
            end_step(step, before)
            return {"step": step, "call": None, "stop": "source_exhausted"}
        if recorder is not None:
            recorder.llm_done(time.perf_counter() - t)
        if call is None:
            # the source logged why (parse_error / api_error): no tool runs
            end_step(step, before)
            return {"step": step, "call": None, "stop": "max_steps" if step >= max_steps else None}
        return {"step": step, "call": call, "stop": None}

    def act(s: LoopState) -> LoopState:
        step, call = s["step"], s["call"]
        before = len(state.history)
        if verbose:
            print("TOOL CALL:\n", call)
        if observer is not None:
            observer.start(step, call)
        t = time.perf_counter()
        try:
            result = execute_tool(call, state)
        except Exception as e:
            result = f"ERROR executing {call.get('tool')}: {e}"
            state.log(call.get("tool", "unknown"), call.get("args", {}), result)
        if recorder is not None:
            recorder.tool_done(call.get("tool", "unknown"), time.perf_counter() - t)
        if verbose:
            print("TOOL RESULT:\n", result)
        end_step(step, before)
        stop = "done" if state.done else "max_steps" if step >= max_steps else None
        return {"call": None, "stop": stop}

    def after_think(s: LoopState) -> str:
        if s.get("stop"):
            return END
        return "act" if s.get("call") is not None else "think"

    def after_act(s: LoopState) -> str:
        return END if s.get("stop") else "think"

    g = StateGraph(LoopState)
    g.add_node("think", think)
    g.add_node("act", act)
    g.add_edge(START, "think")
    g.add_conditional_edges("think", after_think, ["act", "think", END])
    g.add_conditional_edges("act", after_act, ["think", END])
    return g.compile()


def run_graph(state, next_tool_call, max_steps: int, verbose: bool = True, observer=None, recorder=None,
              run_name: str = "chem_llm agent", tags: list[str] | None = None) -> str | None:
    """Run the loop to completion; returns why it stopped."""
    if max_steps < 1:
        return "max_steps"
    graph = build_graph(state, next_tool_call, max_steps=max_steps, verbose=verbose,
                        observer=observer, recorder=recorder)
    final = graph.invoke(
        {"step": 0},
        # two node visits per step, plus slack
        config={"recursion_limit": 2 * max_steps + 10, "run_name": run_name, "tags": tags or []},
    )
    return final.get("stop")
