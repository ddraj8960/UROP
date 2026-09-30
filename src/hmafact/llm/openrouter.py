"""
OpenRouter API Client for HMA-Fact.

Integrates with OpenRouter.ai chat completion API (OpenAI compatible).
Supports Llama 3.1 8B Instruct, Gemini, Claude, and other models.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shelve
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ---------------------------------------------------------------------------
# Disk cache (Issue 5 fix)
# ---------------------------------------------------------------------------
# Cache is keyed on (model, prompt_hash, temperature, max_tokens).
# Disable with OPENROUTER_CACHE=0 in .env or environment.
_CACHE_ENABLED = os.getenv("OPENROUTER_CACHE", "1") == "1"
_CACHE_DIR = Path(".cache/llm_cache")


def _cache_key(model: str, prompt: str, system_prompt: str,
               temperature: float, max_tokens: int) -> str:
    """Stable cache key — SHA256 of all inputs that affect the response."""
    raw = json.dumps({
        "model": model,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> dict | None:
    if not _CACHE_ENABLED:
        return None
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with shelve.open(str(_CACHE_DIR / "responses")) as db:
            entry = db.get(key)
            if entry:
                logger.debug("LLM cache HIT (key=%s...)", key[:12])
            return entry
    except Exception as exc:
        logger.warning("LLM cache read failed: %s", exc)
        return None


def _cache_set(key: str, value: dict) -> None:
    if not _CACHE_ENABLED:
        return
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with shelve.open(str(_CACHE_DIR / "responses")) as db:
            db[key] = value
    except Exception as exc:
        logger.warning("LLM cache write failed: %s", exc)


def generate_response(
    prompt: str,
    system_prompt: str = "You are a concise, accurate assistant. Provide a direct, factual answer.",
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
    retries: int = 3,
) -> dict[str, Any]:
    """
    Send a prompt to OpenRouter API and return the response metadata + text.

    Args:
        prompt: The input user text.
        system_prompt: Optional system instruction.
        model: Model identifier (defaults to OPENROUTER_MODEL env var or llama-3.1-8b-instruct).
        temperature: Sampling temperature (0.0 for deterministic baseline evaluation).
        max_tokens: Maximum output tokens.
        retries: Number of retry attempts on network/rate-limit error.

    Returns:
        dict with keys: 'text', 'model', 'prompt_tokens', 'completion_tokens', 'latency_sec'
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment or .env file.")

    if not model:
        model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")

    # Disk-cache lookup (Issue 5 fix)
    cache_key = _cache_key(model, prompt, system_prompt, temperature, max_tokens)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/hma-fact",
        "X-Title": "HMA-Fact Benchmark Evaluation",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    start_time = time.time()
    last_err: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 429:  # Rate limited
                wait = 2 ** attempt  # exponential backoff (Issue 8 fix)
                logger.warning("OpenRouter rate limited (429). Retrying in %ds...", wait)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            latency = round(time.time() - start_time, 3)
            choices = data.get("choices", [])
            if not choices:
                raise ValueError(f"OpenRouter response contained no choices: {data}")

            text = choices[0]["message"]["content"].strip()
            usage = data.get("usage", {})

            result = {
                "text": text,
                "model": data.get("model", model),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "latency_sec": latency,
            }

            # Store in disk cache before returning (Issue 5 fix)
            _cache_set(cache_key, result)
            return result

        except Exception as e:
            last_err = e
            if attempt < retries:
                wait = 2 ** attempt  # exponential backoff (Issue 8 fix)
                logger.warning("Attempt %d/%d failed (%s). Retrying in %ds...",
                               attempt, retries, e, wait)
                time.sleep(wait)

    raise RuntimeError(f"OpenRouter API call failed after {retries} retries: {last_err}")



class OpenRouterClient:
    """Class wrapper for OpenRouter API client."""

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a concise, accurate assistant. Provide a direct, factual answer.",
        temperature: float = 0.0,
        max_tokens: int = 256,
        retries: int = 3,
    ) -> str:
        """Sends prompt and returns text response string."""
        res = generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
        )
        return res["text"]
