import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, Optional
from openai import OpenAI

# Approximate list prices in USD per 1M tokens.
_PRICING_PER_1M = {
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/text-embedding-3-small": {"input": 0.02, "output": 0.00},
    "deepseek-chat": {"input": 0.14, "output": 0.28},
}

_LOCK = threading.Lock()
_SESSION_TOTALS = {
    "calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "estimated_cost_usd": 0.0,
}

def make_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    """Return an OpenAI-compatible client.
    Can be configured via env vars OPENAI_API_KEY, DEEPSEEK_API_KEY, OPENROUTER_API_KEY, etc.
    """
    key = api_key or os.environ.get("LLM_API_KEY")
    url = base_url or os.environ.get("LLM_BASE_URL")
    provider = os.environ.get("LLM_PROVIDER", "").lower()

    # If no key/url is explicitly set, use provider info or auto-detect
    if not key:
        providers = {
            "openai": {"key_env": "OPENAI_API_KEY", "url": None},
            "deepseek": {"key_env": "DEEPSEEK_API_KEY", "url": "https://api.deepseek.com"},
            "gemini": {"key_env": "GEMINI_API_KEY", "url": "https://generativelanguage.googleapis.com/v1beta/openai/"},
            "openrouter": {"key_env": "OPENROUTER_API_KEY", "url": "https://openrouter.ai/api/v1"},
        }
        
        # If provider is explicitly specified
        if provider in providers:
            info = providers[provider]
            key = os.environ.get(info["key_env"])
            if not url:
                url = info["url"]
        else:
            # Auto-detect by checking available keys in preference order
            for p, info in providers.items():
                if os.environ.get(info["key_env"]):
                    key = os.environ.get(info["key_env"])
                    if not url:
                        url = info["url"]
                    break

    if not key:
        raise RuntimeError("Missing API key for LLM. Set OPENAI_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY, or OPENROUTER_API_KEY.")

    if url:
        return OpenAI(api_key=key, base_url=url, timeout=120.0, max_retries=3)
    return OpenAI(api_key=key, timeout=120.0, max_retries=3)


def _normalize_model(model: str) -> str:
    if not model:
        return ""
    m = model.strip()
    if "/" in m:
        parts = m.split("/")
        if len(parts) == 2:
            provider, name = parts
            if provider in {"openai", "deepseek", "anthropic"}:
                return name if provider != "openai" else m
    return m


def _estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    candidates = [model, _normalize_model(model)]
    pricing = None
    for c in candidates:
        pricing = _PRICING_PER_1M.get(c)
        if pricing:
            break
    if not pricing:
        return None
    return (
        (prompt_tokens * pricing["input"]) / 1_000_000
        + (completion_tokens * pricing["output"]) / 1_000_000
    )


def _extract_usage(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def record_usage(response: Any, model: str, call_type: str = "chat", label: str = "") -> Dict[str, Any]:
    usage = _extract_usage(response)
    prompt_tokens = usage["prompt_tokens"]
    completion_tokens = usage["completion_tokens"]
    total_tokens = usage["total_tokens"]

    est_cost = _estimate_cost_usd(model, prompt_tokens, completion_tokens)

    with _LOCK:
        _SESSION_TOTALS["calls"] += 1
        _SESSION_TOTALS["prompt_tokens"] += prompt_tokens
        _SESSION_TOTALS["completion_tokens"] += completion_tokens
        _SESSION_TOTALS["total_tokens"] += total_tokens
        if est_cost is not None:
            _SESSION_TOTALS["estimated_cost_usd"] += est_cost
        session_snapshot = dict(_SESSION_TOTALS)

    record = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "call_type": call_type,
        "label": label,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": est_cost,
        "session": session_snapshot,
    }

    if os.environ.get("LLM_USAGE_LOG", "1") != "0":
        cost_txt = "n/a" if est_cost is None else f"${est_cost:.6f}"
        print(
            "[llm-usage] "
            f"type={call_type} label={label or '-'} model={model} "
            f"prompt={prompt_tokens} completion={completion_tokens} total={total_tokens} "
            f"cost={cost_txt} session_total=${session_snapshot['estimated_cost_usd']:.6f}"
        )

    return record

def get_session_totals() -> Dict[str, Any]:
    with _LOCK:
        return dict(_SESSION_TOTALS)
