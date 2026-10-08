"""Paths, model names, and decision thresholds."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CHAT_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free").strip()
APP_REFERER = os.getenv("OPENROUTER_REFERER", "http://localhost:8501")
APP_TITLE = "ResolveBot"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION = "northstar_policies"
RETRIEVAL_K = 4
# Cosine similarity floor. MiniLM scores on short FAQ questions often land ~0.27–0.60.
RETRIEVAL_MIN_SCORE = 0.22
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

DATA_DIR = ROOT / "data"
POLICIES_DIR = DATA_DIR / "policies"
CHROMA_DIR = DATA_DIR / "chroma"
ORDERS_PATH = DATA_DIR / "orders.json"
ORDERS_SEED_PATH = DATA_DIR / "orders.seed.json"
TICKETS_DIR = DATA_DIR / "tickets"
AUDIT_DIR = DATA_DIR / "audit"

MAX_RETRIES = 2
LLM_TEMPERATURE = 0.0

COMPANY_NAME = "Northstar Gear"

INTENTS = ("refund", "shipping", "policy", "other")
DECISIONS = ("answer", "tool", "escalate")


def ensure_runtime_dirs() -> None:
    """Create folders the app writes into at runtime."""
    for path in (TICKETS_DIR, AUDIT_DIR, CHROMA_DIR, DATA_DIR):
        path.mkdir(parents=True, exist_ok=True)
