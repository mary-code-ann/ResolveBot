"""Call the mock Order API based on the user message."""

from __future__ import annotations

from resolvebot.audit import append_log, make_entry
from resolvebot.nodes.decide import extract_order_id
from resolvebot.state import AgentState
from resolvebot.tools.order_api import cancel_order, get_order

CANCEL_WORDS = ("cancel", "refund", "return this", "don't want", "do not want")


def _wants_cancel(text: str) -> bool:
    lowered = (text or "").lower()
    return any(word in lowered for word in CANCEL_WORDS)


def call_order_tools(state: AgentState) -> dict:
    text = state.get("user_message") or ""
    order_id = extract_order_id(text)
    if not order_id:
        result = {
            "ok": False,
            "error": "missing_order_id",
            "message": "No order id (ORD-####) was found in the message.",
        }
        return _pack(state, "none", result, "missing order id → escalate")

    try:
        if _wants_cancel(text) or state.get("intent") == "refund":
            tool_name = "cancel_order"
            result = cancel_order(order_id)
        else:
            tool_name = "get_order"
            result = get_order(order_id)
    except Exception as exc:  # noqa: BLE001
        tool_name = "order_api"
        result = {
            "ok": False,
            "error": "tool_exception",
            "message": str(exc),
        }

    if result.get("error") in {"not_found", "tool_exception", "missing_order_id"}:
        next_decision = "escalate"
        reason = result.get("message") or result.get("error")
    else:
        next_decision = "answer"
        reason = f"{tool_name} ok={result.get('ok')}"

    return _pack(state, tool_name, result, reason, next_decision)


def _pack(
    state: AgentState,
    tool_name: str,
    result: dict,
    notes: str,
    next_decision: str | None = None,
) -> dict:
    update = {
        "tool_name": tool_name,
        "tool_result": result,
        "audit_log": append_log(
            state.get("audit_log"),
            make_entry(
                "call_order_tools",
                decision=next_decision or "tool",
                notes=notes,
                extra={"tool": tool_name, "ok": result.get("ok"), "error": result.get("error")},
            ),
        ),
    }
    if next_decision == "escalate":
        update["decision"] = "escalate"
        update["escalation_reason"] = notes
    return update
