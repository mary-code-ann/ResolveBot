"""Assemble the LangGraph workflow."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from resolvebot.nodes.answer import generate_answer
from resolvebot.nodes.classify import classify_intent
from resolvebot.nodes.decide import decide_action
from resolvebot.nodes.escalate import create_human_ticket
from resolvebot.nodes.retrieve import retrieve_policy_node
from resolvebot.nodes.tools_node import call_order_tools
from resolvebot.state import AgentState

_compiled = None


def _route_after_decide(state: AgentState) -> str:
    decision = state.get("decision") or "escalate"
    if decision == "tool":
        return "tools"
    if decision == "answer":
        return "answer"
    return "escalate"


def _route_after_tools(state: AgentState) -> str:
    if state.get("decision") == "escalate":
        return "escalate"
    return "answer"


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("classify", classify_intent)
    workflow.add_node("retrieve", retrieve_policy_node)
    workflow.add_node("decide", decide_action)
    workflow.add_node("tools", call_order_tools)
    workflow.add_node("answer", generate_answer)
    workflow.add_node("escalate", create_human_ticket)

    workflow.add_edge(START, "classify")
    workflow.add_edge("classify", "retrieve")
    workflow.add_edge("retrieve", "decide")
    workflow.add_conditional_edges(
        "decide",
        _route_after_decide,
        {"answer": "answer", "tools": "tools", "escalate": "escalate"},
    )
    workflow.add_conditional_edges(
        "tools",
        _route_after_tools,
        {"answer": "answer", "escalate": "escalate"},
    )
    workflow.add_edge("answer", END)
    workflow.add_edge("escalate", END
    )
    return workflow.compile()


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled
