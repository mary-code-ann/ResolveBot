"""Write a policy-grounded reply. Never invent policy."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from resolvebot.audit import append_log, make_entry
from resolvebot.config import COMPANY_NAME
from resolvebot.llm import get_llm
from resolvebot.state import AgentState

SYSTEM = f"""You are ResolveBot, a support agent for {COMPANY_NAME}.

Hard rules:
- Answer ONLY from the retrieved policy excerpts and any tool JSON provided.
- If the answer is not in those sources, say you cannot confirm it and offer to escalate.
- Never invent return windows, fees, tracking numbers, or refund timelines.
- Cite source filenames in parentheses, e.g. (return_policy.md).
- Be concise (4–8 short sentences). Be warm and specific.
- If tool JSON is present, use those facts for order status.
- Greetings: introduce yourself briefly and say you can help with policy, shipping, and returns.
"""


def _context_block(state: AgentState) -> str:
    parts: list[str] = []
    docs = state.get("retrieved_docs") or []
    if docs:
        parts.append("POLICY EXCERPTS:")
        for i, doc in enumerate(docs, start=1):
            parts.append(f"[{i} {doc.get('source')}] {doc.get('content')}")
    tool = state.get("tool_result")
    if tool:
        parts.append(f"TOOL RESULT ({state.get('tool_name') or 'order_api'}): {tool}")
    if not parts:
        parts.append("No policy excerpts and no tool result.")
    return "\n\n".join(parts)


def generate_answer(state: AgentState) -> dict:
    llm = get_llm()
    user = state.get("user_message") or ""
    prompt = f"Customer message:\n{user}\n\n{_context_block(state)}"
    try:
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM),
                HumanMessage(content=prompt),
            ]
        )
        text = str(response.content).strip()
        notes = "grounded answer generated"
    except Exception as exc:  # noqa: BLE001
        text = (
            "I hit a temporary model error and could not finish a grounded answer. "
            "Please try again, or ask me to escalate to a human."
        )
        notes = f"answer failed: {exc}"

    return {
        "final_answer": text,
        "escalated": False,
        "messages": [AIMessage(content=text)],
        "audit_log": append_log(
            state.get("audit_log"),
            make_entry(
                "generate_answer",
                decision="answer",
                confidence=state.get("confidence"),
                notes=notes,
                extra={"sources": state.get("sources") or []},
            ),
        ),
    }
