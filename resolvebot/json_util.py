"""Parse JSON from messy LLM replies and retry failed calls."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from resolvebot.config import MAX_RETRIES

_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of model text (raw or fenced)."""
    if not text or not str(text).strip():
        raise ValueError("Empty model response")
    blob = str(text).strip()
    fenced = _FENCE.search(blob)
    if fenced:
        blob = fenced.group(1).strip()
    start = blob.find("{")
    end = blob.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response")
    return json.loads(blob[start : end + 1])


def invoke_json(fn: Callable[[], Any], retries: int = MAX_RETRIES) -> dict[str, Any]:
    """Call `fn` until it returns parseable JSON, then return the dict."""
    last_error: Exception | None = None
    attempts = retries + 1
    for _ in range(attempts):
        try:
            result = fn()
            if isinstance(result, dict):
                return result
            content = getattr(result, "content", result)
            return extract_json(str(content))
        except Exception as exc:  # noqa: BLE001 — we retry any parse/call failure
            last_error = exc
    raise last_error or RuntimeError("JSON invoke failed")
