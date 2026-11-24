"""Thin wrapper around the DeepSeek chat completions API (OpenAI-compatible).

Every call returns the text plus token usage and latency, which is exactly the
data the evaluation stage needs to compare the three input strategies on cost
and time.
"""
from __future__ import annotations

import time

from openai import OpenAI

import config

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY is not set. Copy .env.example to .env and fill it in.")
        _client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
    return _client


def chat(messages: list[dict], max_tokens: int, temperature: float | None = None) -> dict:
    client = get_client()
    resp = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=config.TEMPERATURE if temperature is None else temperature,
    )
    choice = resp.choices[0]
    return {
        "content": (choice.message.content or "").strip(),
        "model": resp.model,
        "finish_reason": choice.finish_reason,
        "prompt_tokens": int(resp.usage.prompt_tokens),
        "completion_tokens": int(resp.usage.completion_tokens),
        "total_tokens": int(resp.usage.total_tokens),
    }


def chat_with_retry(messages: list[dict], max_tokens: int,
                    retries: int = 4, backoff: float = 2.0,
                    temperature: float | None = None) -> dict:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            return chat(messages, max_tokens, temperature=temperature)
        except Exception as e:  # noqa: BLE001 — retry transient API/network errors
            last_err = e
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")
