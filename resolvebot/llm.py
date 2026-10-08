"""OpenRouter chat client (OpenAI-compatible)."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from resolvebot.config import (
    APP_REFERER,
    APP_TITLE,
    CHAT_MODEL,
    LLM_TEMPERATURE,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)


def get_llm(temperature: float | None = None) -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Copy .env.example to .env and add your key."
        )
    return ChatOpenAI(
        model=CHAT_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=LLM_TEMPERATURE if temperature is None else temperature,
        default_headers={
            "HTTP-Referer": APP_REFERER,
            "X-Title": APP_TITLE,
        },
    )
