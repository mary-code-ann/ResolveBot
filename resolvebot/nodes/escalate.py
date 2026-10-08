"""Create a human ticket and a short handoff summary."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from langchain_core.messages import AIMessage

from resolvebot.audit import append_log, make_entry, utc_now
from resolvebot.config import TICKETS_DIR, ensure_runtime_dirs
from resolvebot.state import AgentState


def _next_ticket_id() -> str:
    ensure_runtime_dirs()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    existing = list(TICKETS_DIR.glob("TICKET-*.json"))
    return f"TICKET-{stamp}-{len(existing) + 1:03d}"


def _summarize(state: AgentState) -> str:
    docs = state.get("retrieved_docs") or []
    top = docs[0]["content"][:240].replace("\n", " ") if docs else "no policy hit"
    tool = state.get("tool_result")
    tool_line = ""
    if tool:
        tool_line = f" Tool={state.get('tool_name')} ok={tool.get('ok')} error={tool.get('error')}."
    return (
        f"Intent={state.get('intent')}. Reason={state.get('escalation_reason')}. "
        f"Best retrieval={state.get('retrieval_score')}. Sources={state.get('sources')}. "
        f"Top excerpt: {top}.{tool_line}"
    )


def create_human_ticket(state: AgentState) -> dict:
    ticket_id = _next_ticket_id()
    reason = state.get("escalation_reason") or "low confidence or policy gap"
    summary = _summarize(state)
    payload = {
        "ticket_id": ticket_id,
        "created_at": utc_now(),
        "status": "open",
        "reason": reason,
        "intent": state.get("intent"),
        "user_message": state.get("user_message"),
        "retrieval_score": state.get("retrieval_score"),
        "sources": state.get("sources") or [],
        "tool_result": state.get("tool_result"),
        "summary": summary,
        "audit_log": state.get("audit_log") or [],
    }
    path = TICKETS_DIR / f"{ticket_id}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    answer = (
        f"I’m escalating this to a human specialist and created **{ticket_id}**. "
        f"Reason: {reason}. A teammate will pick this up from the human queue. "
        "I have attached a short summary of this thread for them."
    )
    return {
        "ticket_id": ticket_id,
        "escalated": True,
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "audit_log": append_log(
            state.get("audit_log"),
            make_entry(
                "create_human_ticket",
                decision="escalate",
                confidence=state.get("confidence"),
                notes=f"{ticket_id}: {reason}",
                extra={"ticket_id": ticket_id, "path": str(path)},
            ),
        ),
    }


def list_tickets() -> list[dict]:
    ensure_runtime_dirs()
    tickets = []
    for path in sorted(TICKETS_DIR.glob("TICKET-*.json"), reverse=True):
        try:
            tickets.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return tickets
