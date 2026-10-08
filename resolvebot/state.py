"""Shared LangGraph state for one support turn."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_message: str
    intent: str
    retrieved_docs: list
    retrieval_score: float
    decision: str
    confidence: float
    tool_name: str
    tool_result: dict | None
    retry_count: int
    escalated: bool
    ticket_id: str | None
    escalation_reason: str
    sources: list
    audit_log: list
    final_answer: str


def empty_state(user_message: str) -> dict[str, Any]:
    """Defaults for a new graph run."""
    return {
        "messages": [{"role": "user", "content": user_message}],
        "user_message": user_message,
        "intent": "",
        "retrieved_docs": [],
        "retrieval_score": 0.0,
        "decision": "",
        "confidence": 0.0,
        "tool_name": "",
        "tool_result": None,
        "retry_count": 0,
        "escalated": False,
        "ticket_id": None,
        "escalation_reason": "",
        "sources": [],
        "audit_log": [],
        "final_answer": "",
    }
