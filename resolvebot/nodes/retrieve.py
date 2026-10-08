"""Retrieve policy chunks. No LLM call."""

from __future__ import annotations

from resolvebot.audit import append_log, make_entry
from resolvebot.rag.retriever import retrieve_policy
from resolvebot.state import AgentState

_INTENT_HINTS = {
    "refund": "return refund cancel policy",
    "shipping": "shipping tracking delivery package",
    "policy": "company policy FAQ warranty returns",
    "other": "",
}


def retrieve_policy_node(state: AgentState) -> dict:
    message = state.get("user_message") or ""
    hint = _INTENT_HINTS.get(state.get("intent") or "", "")
    query = f"{hint} {message}".strip()
    docs, score = retrieve_policy(query)
    sources = []
    seen = set()
    for doc in docs:
        name = doc.get("source") or "unknown"
        if name not in seen:
            seen.add(name)
            sources.append(name)

    return {
        "retrieved_docs": docs,
        "retrieval_score": score,
        "sources": sources,
        "audit_log": append_log(
            state.get("audit_log"),
            make_entry(
                "retrieve_policy",
                decision="retrieve",
                confidence=score,
                notes=f"{len(docs)} chunks, best_score={score}",
                extra={"sources": sources},
            ),
        ),
    }
