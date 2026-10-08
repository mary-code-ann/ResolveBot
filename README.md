# ResolveBot

Policy-grounded support agent with tools and human handoff.

**Interview one-liner:** “I built a LangGraph agent that answers support tickets only from company policy (RAG), can call order/refund APIs as tools, and escalates when confidence is low — with retries and an audit trail.”

```
User message
    → Classify intent (refund / shipping / policy / other)
    → RAG retrieve (Northstar Gear FAQs + policies)
    → Decide: answer | call tool | escalate
        → Tool (mock Order API) → grounded final answer
        → Escalate → human ticket + thread summary
```

This is a **workflow graph**, not a free-roaming agent. You control the path.

## What it demonstrates

- **LangGraph state** — each node updates a shared `AgentState` (`intent`, docs, decision, confidence, audit log)
- **RAG grounding** — answers must come from retrieved policy chunks; the model is told not to invent policy
- **Hybrid routing** — deterministic rules first (order id, legal language, weak retrieval), LLM only for gray areas
- **Tools** — `get_order` / `cancel_order` against a local JSON “API”
- **Human handoff** — writes `data/tickets/TICKET-*.json` when confidence is low or tools miss
- **Retries + audit** — JSON parse and tool I/O retry twice, then escalate; every node appends a log line

## Stack

| Piece | Choice | Why |
| --- | --- | --- |
| UI | Streamlit | Fast to demo |
| Orchestration | LangGraph | Explicit nodes and conditional edges |
| Chat model | OpenRouter free models (`openrouter/free`) | No OpenAI bill |
| Embeddings | Local `all-MiniLM-L6-v2` | Saves the 50 free LLM requests/day |
| Vector store | Chroma on disk | No cloud database |

## Setup

You need **Python 3.11+** and a free [OpenRouter API key](https://openrouter.ai/keys).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and paste your key:

```
OPENROUTER_API_KEY=sk-or-v1-your-new-key-here
OPENROUTER_MODEL=openrouter/free
```

Never commit `.env`. If a key was pasted into chat, revoke it at [openrouter.ai/keys](https://openrouter.ai/keys) and create a new one.

Build the policy index (first run downloads ~80MB for the embedding model):

```powershell
python scripts/ingest.py
```

Start the chat UI:

```powershell
streamlit run app.py
```

OpenRouter free models are typically **20 requests/minute** and about **50 requests/day** if you have never bought credits. Local embeddings keep indexing off that quota.

## 5-minute demo script

Use the buttons in the UI or type these:

1. **“What’s your return window?”** — policy answer, cites `return_policy.md`
2. **“Where is order ORD-1002?”** — `get_order` tool, shipped / not refundable yet
3. **“Cancel ORD-1001”** — delivered 5 days ago, cancel succeeds
4. **“Cancel ORD-1003”** — delivered 45 days ago, refused with policy citation
5. **“This is fraud, I want a lawyer”** — escalate; a ticket appears in the Human queue
6. Open the **Audit trail** and walk classify → retrieve → decide → answer/tool/escalate

Reset demo data with **Reset demo orders** after you cancel ORD-1001.

Unknown ids such as `ORD-1005` miss in the mock API and escalate.

## Project layout

```
app.py                      Streamlit UI
scripts/ingest.py           Build the Chroma index
resolvebot/
  graph.py                  LangGraph wiring
  state.py                  Shared state
  llm.py                    OpenRouter client
  nodes/                    classify, retrieve, decide, tools, answer, escalate
  rag/                      local embeddings + Chroma
  tools/order_api.py        mock get_order / cancel_order
  audit.py                  JSONL trail
data/policies/              Northstar Gear markdown
data/orders.json            Working mock orders
data/orders.seed.json       Reset snapshot
data/tickets/               Human tickets (runtime)
data/audit/                 Daily JSONL audit files
```

## Interview talking points

**Why a graph instead of one giant prompt?**  
Classification, retrieval, routing, tools, and answering are different jobs. Separate nodes make failures visible and let you swap a node without rewriting the whole agent.

**Why hybrid decide (rules + LLM)?**  
Legal/fraud language and “talk to a human” should never depend on a flaky free model. Rules handle the safety cases; the LLM only routes gray areas.

**Why local embeddings + cloud LLM?**  
Embedding every policy chunk through OpenRouter would burn the free daily limit before you finish a demo. Retrieval is local and free; the LLM is reserved for classify / decide / answer.

**How does human handoff work?**  
Escalate when retrieval is weak, intent is `other`, the customer asks for a person, legal words appear, or a tool returns `not_found` after retries. The escalate node writes a ticket JSON plus a thread summary.

**What do retries and the audit trail look like in production?**  
Bad JSON from the model and mock API I/O retry twice, then escalate. Every node appends `{time, node, decision, confidence, notes}`. Streamlit shows the in-memory trail; `data/audit/` keeps JSONL.

**Honest limits**  
`openrouter/free` may pick a different model per request. Free-model JSON quality varies — that is why we parse defensively and prefer rules. This is a demo, not a production helpdesk.

## Next steps you can mention

- Swap the JSON file for a real order API
- Add PDF loaders (same ingest path)
- LangSmith tracing
- A supervisor multi-agent if the desk grows beyond one workflow
