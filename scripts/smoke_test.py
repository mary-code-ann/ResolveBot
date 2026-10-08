"""Offline smoke test for RAG, mock tools, and rule routing (no LLM)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resolvebot.graph import get_graph
from resolvebot.nodes.decide import decide_action, extract_order_id
from resolvebot.rag.retriever import retrieve_policy
from resolvebot.tools.order_api import cancel_order, get_order, reset_orders


def main() -> None:
    docs, score = retrieve_policy("What is the return window?")
    print("RETRIEVE", score, [d["source"] for d in docs[:3]])
    print("SNIP", docs[0]["content"][:80].replace("\n", " "))

    lookup = get_order("ORD-1002")
    print("GET", lookup["ok"], lookup["order"]["status"])
    print("MISS", get_order("ORD-1005")["error"])
    old = cancel_order("ORD-1003")
    print("CANCEL_OLD", old["ok"], old.get("error"))
    new = cancel_order("ORD-1001")
    print("CANCEL_NEW", new["ok"], new.get("error"))
    reset_orders()

    legal = decide_action(
        {
            "user_message": "This is fraud, I want a lawyer",
            "intent": "refund",
            "retrieval_score": 0.6,
            "audit_log": [],
        }
    )
    print("DECIDE_LEGAL", legal["decision"])

    tool = decide_action(
        {
            "user_message": "Where is order ORD-1002?",
            "intent": "shipping",
            "retrieval_score": 0.7,
            "audit_log": [],
        }
    )
    print("DECIDE_TOOL", tool["decision"], extract_order_id("Where is order ORD-1002?"))

    answer = decide_action(
        {
            "user_message": "What's your return window?",
            "intent": "policy",
            "retrieval_score": 0.7,
            "audit_log": [],
        }
    )
    print("DECIDE_ANS", answer["decision"])
    print("GRAPH", type(get_graph()).__name__)


if __name__ == "__main__":
    main()
