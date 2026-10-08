"""One-turn helper used by Streamlit and smoke tests."""

from __future__ import annotations

from typing import Any

from resolvebot.audit import persist_audit
from resolvebot.graph import get_graph
from resolvebot.state import empty_state


def run_turn(user_message: str) -> dict[str, Any]:
    graph = get_graph()
    result = graph.invoke(empty_state(user_message))
    persist_audit(result.get("audit_log") or [], user_message)
    return result
