"""Streamlit chat UI for ResolveBot."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from resolvebot.config import CHAT_MODEL, COMPANY_NAME, OPENROUTER_API_KEY, ensure_runtime_dirs
from resolvebot.nodes.escalate import list_tickets
from resolvebot.rag.ingest import ingest, index_exists
from resolvebot.run import run_turn
from resolvebot.tools.order_api import reset_orders

DEMO_PROMPTS = [
    "What's your return window?",
    "Where is order ORD-1002?",
    "Cancel ORD-1001",
    "Cancel ORD-1003",
    "This is fraud, I want a lawyer",
]


def _boot() -> None:
    ensure_runtime_dirs()
    if not index_exists():
        with st.spinner("Building the local policy index (first run downloads a small embedding model)..."):
            ingest(rebuild=False)


st.set_page_config(page_title="ResolveBot", page_icon="🧭", layout="wide")
_boot()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

st.title("ResolveBot")
st.caption(
    f"Policy-grounded support agent for {COMPANY_NAME} · LangGraph · RAG · tools · human handoff"
)

if not OPENROUTER_API_KEY:
    st.error(
        "OPENROUTER_API_KEY is missing. Copy `.env.example` to `.env`, add a key from "
        "https://openrouter.ai/keys, then restart Streamlit."
    )

left, right = st.columns([1.6, 1])

with right:
    st.subheader("This turn")
    result = st.session_state.last_result
    if not result:
        st.info("Send a message to see intent, sources, and the audit trail.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Intent", result.get("intent") or "—")
        c2.metric("Decision", result.get("decision") or "—")
        c3.metric("Confidence", f"{float(result.get('confidence') or 0):.2f}")
        st.metric("Retrieval score", f"{float(result.get('retrieval_score') or 0):.2f}")
        if result.get("escalated"):
            st.warning(f"Escalated · {result.get('ticket_id')}")
        sources = result.get("sources") or []
        st.markdown("**Sources**")
        st.write(", ".join(sources) if sources else "None")
        if result.get("tool_result"):
            st.markdown(f"**Tool:** `{result.get('tool_name')}`")
            st.json(result.get("tool_result"))
        with st.expander("Audit trail", expanded=True):
            for step in result.get("audit_log") or []:
                st.markdown(
                    f"- `{step.get('time')}` **{step.get('node')}** → "
                    f"{step.get('decision') or '—'} · {step.get('notes')}"
                )

    st.divider()
    st.subheader("Human queue")
    tickets = list_tickets()
    if not tickets:
        st.caption("No open tickets yet.")
    for ticket in tickets[:8]:
        with st.expander(f"{ticket.get('ticket_id')} · {ticket.get('reason')}"):
            st.write(ticket.get("summary"))
            st.caption(ticket.get("user_message"))

    st.divider()
    if st.button("Rebuild policy index"):
        with st.spinner("Re-indexing markdown policies..."):
            info = ingest(rebuild=True)
        st.success(f"Indexed {info['chunks']} chunks from {len(info['files'])} files.")
    if st.button("Reset demo orders"):
        reset_orders()
        st.success("orders.json restored from the seed file.")

with left:
    st.markdown("**Try a demo prompt**")
    clicked = None
    for row in (DEMO_PROMPTS[:3], DEMO_PROMPTS[3:]):
        cols = st.columns(len(row))
        for col, prompt in zip(cols, row):
            if col.button(prompt, use_container_width=True):
                clicked = prompt

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    typed = st.chat_input("Ask about returns, shipping, or an order id like ORD-1001")
    incoming = clicked or typed

    if incoming:
        st.session_state.messages.append({"role": "user", "content": incoming})
        if not OPENROUTER_API_KEY:
            reply = "Add OPENROUTER_API_KEY to `.env` and restart the app."
        else:
            with st.spinner(f"Running the graph · model `{CHAT_MODEL}`"):
                try:
                    turn = run_turn(incoming)
                    st.session_state.last_result = turn
                    reply = turn.get("final_answer") or "I could not produce an answer."
                except Exception as exc:  # noqa: BLE001
                    reply = f"The graph failed: {exc}"
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()
