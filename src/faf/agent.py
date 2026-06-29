import json
import os
from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, END

LLM_MODEL = os.environ.get("LLM_MODEL") or os.environ.get("CI_AGENT_MODEL") or "gpt-4o-mini"

from faf.models import RunEvent
from faf.bayesian_tracker import registry, BeliefState
from faf.retrieval_prior import get_retrieval_prior
from faf.tool_selection import rank_tools_by_eig, format_eig_for_prompt
from faf.llm import make_client, record_usage
from faf.prompts import PLANNER_PROMPT, CLASSIFY_PROMPT

from faf.tools.common import deep_log_analysis, search_similar_failures, search_web_for_error
from faf.tools.github import inspect_commit_diff, check_run_history, inspect_workflow_file, inspect_pr_context, inspect_dependency_changes
from faf.tools.kubernetes import inspect_k8s_events

MAX_STEPS = 5
ENTROPY_THRESHOLD = 0.5

def merge_lists(a: list, b: list) -> list:
    if a is None: a = []
    if b is None: b = []
    return a + b

class AgentState(TypedDict):
    event: RunEvent
    beliefs: Dict[str, float]
    belief_history: Annotated[list, merge_lists]
    confidence: float
    entropy: float
    
    current_step: int
    tools_available: list
    _next_tool: str
    done: bool

    investigation_log: Annotated[list, merge_lists]
    tools_called: Annotated[list, merge_lists]
    classification: Dict[str, Any]

def initialize(state: AgentState) -> dict:
    event = state["event"]
    error_lines = [step.error_text for step in event.failed_steps if step.error_text]
    
    prior_probs, similar_cases = get_retrieval_prior(
        commit_title=event.metadata.get("commit_title", ""),
        error_lines=error_lines,
        mentioned_files=[]
    )
    
    tracker = BeliefState(probabilities=prior_probs)
    signal_results = registry.evaluate_available(event)
    for sig_name, likelihoods in signal_results.items():
        tracker.update(likelihoods, signal_name=sig_name)
    
    tools = ["deep_log_analysis", "search_similar_failures"]
    if event.source == "github":
        tools.extend(["inspect_commit_diff", "check_run_history", "inspect_workflow_file", "inspect_dependency_changes"])
    elif event.source == "kubernetes":
        tools.extend(["inspect_k8s_events"])
        
    return {
        "beliefs": tracker.probabilities,
        "belief_history": tracker.history,
        "confidence": tracker.confidence(),
        "entropy": tracker.entropy(),
        "current_step": 0,
        "tools_available": tools,
        "done": False,
        "investigation_log": [f"Preprocessed {event.source} event. Confidence: {tracker.confidence():.1%}"],
        "tools_called": [],
        "classification": {}
    }

def planner(state: AgentState) -> dict:
    step = state.get("current_step", 0) + 1
    if step > MAX_STEPS:
        return {"done": True, "current_step": step, "_next_tool": "classify"}

    tracker = BeliefState(probabilities=state["beliefs"])
    available = state.get("tools_available", [])
    
    if not available or state.get("entropy", 1.0) <= ENTROPY_THRESHOLD:
        return {"done": True, "current_step": step, "_next_tool": "classify"}

    rankings = rank_tools_by_eig(tracker, available)
    eig_table = format_eig_for_prompt(rankings)
    
    prompt = PLANNER_PROMPT.format(
        step=step, max_steps=MAX_STEPS,
        entropy=state.get("entropy", 1.0),
        top_beliefs=", ".join(f"{c} ({p:.0%})" for c,p in tracker.top_n(3)),
        tools_used=", ".join(state.get("tools_called", [])) or "none",
        tools_available=", ".join(available),
        eig_table=eig_table,
        investigation_log="\\n".join(state.get("investigation_log", [])[-5:]),
        threshold=ENTROPY_THRESHOLD
    )
    
    try:
        client = make_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        record_usage(response, LLM_MODEL, "chat", "planner")
        result = json.loads(response.choices[0].message.content)
        next_tool = result.get("tool", "classify")
        reasoning = result.get("reasoning", "")
    except Exception as e:
        next_tool = "classify"
        reasoning = f"Planner LLM failed: {e}"

    if next_tool not in available and next_tool != "classify":
        next_tool = "classify"
        
    return {
        "current_step": step,
        "_next_tool": next_tool,
        "investigation_log": [f"[planner] Chose {next_tool} because: {reasoning}"]
    }

def classify(state: AgentState) -> dict:
    tracker = BeliefState(probabilities=state["beliefs"])
    event = state["event"]
    
    prompt = CLASSIFY_PROMPT.format(
        preprocessing_summary=json.dumps(event.metadata),
        investigation_log="\\n".join(state.get("investigation_log", [])),
        beliefs=", ".join(f"{c}: {p:.1%}" for c, p in tracker.probabilities.items() if p > 0.05),
        error_lines="\\n".join([step.error_text for step in event.failed_steps if step.error_text][:1]),
        changed_files="(mocked)",
        commit_diff="(mocked)",
        failed_step_context=str(event.failed_steps[0].metadata if event.failed_steps else {}),
        dependency_changes="(mocked)",
        runner_environment="(mocked)",
        pr_context="(mocked)",
        workflow_signals="(mocked)",
        run_history="(mocked)",
        similar_failures="(mocked)",
        source=event.source,
        run_id=event.run_id,
        failed=event.metadata.get("jobs_failed", len(event.failed_steps)),
        total=event.metadata.get("jobs_total", 1),
        categories=", ".join(tracker.probabilities.keys())
    )
    
    try:
        client = make_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        record_usage(response, LLM_MODEL, "chat", "classify")
        result = json.loads(response.choices[0].message.content)
        result["confidence"] = result.get("confidence", tracker.confidence())
        result["probability"] = tracker.probabilities.get(result.get("category", "UNKNOWN"), 0.0)
    except Exception as e:
        top_cat = max(tracker.probabilities.items(), key=lambda x: x[1])
        result = {
            "category": top_cat[0],
            "probability": top_cat[1],
            "confidence": tracker.confidence(),
            "reasoning": f"LLM parsing failed. Fallback to bayesian max: {e}"
        }
    
    return {"classification": result}

def route_after_planner(state: AgentState) -> str:
    if state.get("done", False):
        return "classify"
    return state.get("_next_tool", "classify")

def _wrap_tool(tool_func):
    def wrapper(state: AgentState):
        result = tool_func(state)
        available = [t for t in state.get("tools_available", []) if t != tool_func.__name__]
        result["tools_available"] = available
        if "tools_called" not in result:
            result["tools_called"] = [tool_func.__name__]
        return result
    return wrapper

def build_agent_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    
    graph.add_node("initialize", initialize)
    graph.add_node("planner", planner)
    graph.add_node("deep_log_analysis", _wrap_tool(deep_log_analysis))
    graph.add_node("search_similar_failures", _wrap_tool(search_similar_failures))
    graph.add_node("inspect_commit_diff", _wrap_tool(inspect_commit_diff))
    graph.add_node("check_run_history", _wrap_tool(check_run_history))
    graph.add_node("inspect_workflow_file", _wrap_tool(inspect_workflow_file))
    graph.add_node("inspect_dependency_changes", _wrap_tool(inspect_dependency_changes))
    graph.add_node("inspect_k8s_events", _wrap_tool(inspect_k8s_events))
    graph.add_node("classify", classify)

    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "planner")

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "deep_log_analysis": "deep_log_analysis",
            "search_similar_failures": "search_similar_failures",
            "inspect_commit_diff": "inspect_commit_diff",
            "check_run_history": "check_run_history",
            "inspect_workflow_file": "inspect_workflow_file",
            "inspect_dependency_changes": "inspect_dependency_changes",
            "inspect_k8s_events": "inspect_k8s_events",
            "classify": "classify",
        }
    )

    for tool in ["deep_log_analysis", "search_similar_failures", "inspect_commit_diff", "check_run_history", "inspect_workflow_file", "inspect_dependency_changes", "inspect_k8s_events"]:
        graph.add_edge(tool, "planner")

    graph.add_edge("classify", END)

    return graph.compile()

def run_agent(event: RunEvent) -> dict:
    """
    Main entry point for the analysis loop.
    Invokes the LangGraph state machine.
    """
    graph = build_agent_graph()
    initial_state = {"event": event}
    final_state = graph.invoke(initial_state)
    return final_state
