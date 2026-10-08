"""Classify a support message into refund / shipping / policy / other."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from resolvebot.audit import append_log, make_entry
from resolvebot.config import COMPANY_NAME, INTENTS
from resolvebot.json_util import invoke_json
from resolvebot.llm import get_llm
from resolvebot.state import AgentState

SYSTEM = f"""You classify customer messages for {COMPANY_NAME} support.
Return ONLY a JSON object with this shape:
{{"intent": "refund"|"shipping"|"policy"|"other", "confidence": 0.0}}

Definitions:
- refund: cancel an order, get money back, return an item they already have
- shipping: where is my package, tracking, delivery date, change address
- policy: return window, warranty, restocking, how refunds work, general FAQ
- other: greetings, thanks, off-topic, or truly unclear

confidence is between 0 and 1.
Do not add markdown or extra keys."""


def classify_intent(state: AgentState) -> dict:
    message = state.get("user_message") or ""
    llm = get_llm()

    def _call():
        return llm.invoke(
            [
                SystemMessage(content=SYSTEM),
                HumanMessage(content=message),
            ]
        )

    try:
        parsed = invoke_json(_call)
        intent = str(parsed.get("intent", "other")).lower().strip()
        if intent not in INTENTS:
            intent = "other"
        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        notes = "classified from model JSON"
    except Exception as exc:  # noqa: BLE001
        intent = "other"
        confidence = 0.2
        notes = f"classify failed after retries: {exc}"

    return {
        "intent": intent,
        "confidence": confidence,
        "audit_log": append_log(
            state.get("audit_log"),
            make_entry(
                "classify_intent",
                decision=intent,
                confidence=confidence,
                notes=notes,
            ),
        ),
    }
