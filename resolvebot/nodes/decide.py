"""Hybrid router: deterministic rules first, LLM only for gray areas."""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from resolvebot.audit import append_log, make_entry
from resolvebot.config import RETRIEVAL_MIN_SCORE
from resolvebot.json_util import invoke_json
from resolvebot.llm import get_llm
from resolvebot.state import AgentState

ORDER_RE = re.compile(r"\bORD-\d{4}\b", re.IGNORECASE)

HUMAN_PHRASES = (
    "speak to a human",
    "talk to a human",
    "talk to someone",
    "real person",
    "customer service rep",
    "manager",
    "supervisor",
    "agent please",
)

LEGAL_PHRASES = (
    "lawyer",
    "attorney",
    "sue",
    "lawsuit",
    "legal action",
    "chargeback",
    "bank dispute",
    "fraud",
    "scam",
    "better business bureau",
    "attorney general",
)

GREETINGS = {
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "good morning",
    "good afternoon",
    "good evening",
    "yo",
}


def extract_order_id(text: str) -> str | None:
    match = ORDER_RE.search(text or "")
    return match.group(0).upper() if match else None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _is_greeting(text: str) -> bool:
    cleaned = re.sub(r"[^a-z\s]", "", _norm(text))
    return cleaned in GREETINGS or cleaned.startswith("hi ") or cleaned.startswith("hello ")


def _wants_human(text: str) -> bool:
    lowered = _norm(text)
    return any(phrase in lowered for phrase in HUMAN_PHRASES)


def _legal_or_angry(text: str) -> bool:
    lowered = _norm(text)
    return any(phrase in lowered for phrase in LEGAL_PHRASES)


def _needs_order_tool(intent: str, text: str) -> bool:
    if not extract_order_id(text):
        return False
    if intent in {"refund", "shipping"}:
        return True
    lowered = _norm(text)
    action_words = ("cancel", "refund", "return", "where is", "track", "status", "ship")
    return any(word in lowered for word in action_words)


def _rule_decision(state: AgentState) -> tuple[str, float, str] | None:
    text = state.get("user_message") or ""
    intent = state.get("intent") or "other"
    score = float(state.get("retrieval_score") or 0.0)

    if _wants_human(text):
        return "escalate", 0.95, "customer asked for a human"
    if _legal_or_angry(text):
        return "escalate", 0.95, "legal, fraud, or chargeback language"
    if intent == "other" and _is_greeting(text):
        return "answer", 0.9, "greeting — no tools or policy needed"
    if intent == "other":
        return "escalate", 0.7, "intent classified as other"
    if _needs_order_tool(intent, text):
        return "tool", 0.9, "order id present and action looks operational"
    if score < RETRIEVAL_MIN_SCORE:
        return "escalate", 0.75, f"weak retrieval score {score:.2f}"
    if intent in {"policy", "refund", "shipping"}:
        return "answer", 0.8, "policy-grounded answer without a tool"
    return None


DECIDE_SYSTEM = """You are the router for a support agent.
Choose the next action. Return ONLY JSON:
{"decision": "answer"|"tool"|"escalate", "confidence": 0.0, "reason": "short"}

Rules:
- tool: only if the user gave an order id like ORD-1001 AND they want status, cancel, or refund
- escalate: human requested, legal/fraud, or you cannot help from policy
- answer: policy / FAQ questions, or ask them for an order id
"""


def _llm_decision(state: AgentState) -> tuple[str, float, str]:
    llm = get_llm()
    payload = (
        f"intent={state.get('intent')}\n"
        f"retrieval_score={state.get('retrieval_score')}\n"
        f"message={state.get('user_message')}\n"
    )

    def _call():
        return llm.invoke(
            [
                SystemMessage(content=DECIDE_SYSTEM),
                HumanMessage(content=payload),
            ]
        )

    parsed = invoke_json(_call)
    decision = str(parsed.get("decision", "escalate")).lower().strip()
    if decision not in {"answer", "tool", "escalate"}:
        decision = "escalate"
    confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
    reason = str(parsed.get("reason") or "llm router")
    return decision, confidence, reason


def decide_action(state: AgentState) -> dict:
    ruled = _rule_decision(state)
    source = "rules"
    if ruled:
        decision, confidence, reason = ruled
    else:
        source = "llm"
        try:
            decision, confidence, reason = _llm_decision(state)
        except Exception as exc:  # noqa: BLE001
            decision, confidence, reason = "escalate", 0.4, f"router failed: {exc}"

    return {
        "decision": decision,
        "confidence": confidence,
        "escalation_reason": reason if decision == "escalate" else state.get("escalation_reason", ""),
        "audit_log": append_log(
            state.get("audit_log"),
            make_entry(
                "decide_action",
                decision=decision,
                confidence=confidence,
                notes=f"{source}: {reason}",
            ),
        ),
    }
