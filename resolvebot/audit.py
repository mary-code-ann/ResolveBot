"""In-memory audit entries plus JSONL persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from resolvebot.config import AUDIT_DIR, ensure_runtime_dirs


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_entry(
    node: str,
    *,
    decision: str | None = None,
    confidence: float | None = None,
    notes: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "time": utc_now(),
        "node": node,
        "decision": decision,
        "confidence": confidence,
        "notes": notes,
    }
    if extra:
        entry["extra"] = extra
    return entry


def append_log(audit_log: list[dict[str, Any]] | None, entry: dict[str, Any]) -> list[dict[str, Any]]:
    log = list(audit_log or [])
    log.append(entry)
    return log


def persist_audit(audit_log: list[dict[str, Any]], user_message: str) -> Path:
    """Append one conversation turn's audit trail to today's JSONL file."""
    ensure_runtime_dirs()
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = AUDIT_DIR / f"{day}.jsonl"
    record = {
        "time": utc_now(),
        "user_message": user_message,
        "steps": audit_log,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
