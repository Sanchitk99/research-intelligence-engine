"""Automatic failover, retry policy, and provider chain resolution for LLM generation."""

import logging
import os
import random
import re
import time
from typing import List, Dict, Any, Optional, Tuple, Union
import httpx

from src.generation.providers.base import (
    BaseLLMProvider,
    ProviderError,
    ProviderAuthError,
    ProviderTimeoutError,
    RateLimitError,
)
from src.generation.providers.factory import get_llm_provider

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
NON_RETRYABLE_CLIENT_CODES = {400, 422}
AUTH_STATUS_CODES = {401, 403}


def sanitize_error_message(msg: str) -> str:
    """Strip API keys, authorization tokens, and query secrets from error messages."""
    if not msg:
        return ""
    # Strip bearer tokens
    s = re.sub(r"(bearer\s+)[a-zA-Z0-9_\-\.]{8,}", r"\1[REDACTED]", msg, flags=re.IGNORECASE)
    # Strip key params e.g. key=... or api_key=...
    s = re.sub(r"((?:api[_-]?key|key|token)=)[a-zA-Z0-9_\-\.]{8,}", r"\1[REDACTED]", s, flags=re.IGNORECASE)
    # Strip x-goog-api-key
    s = re.sub(r"(x-goog-api-key[:=]\s*)[a-zA-Z0-9_\-\.]{8,}", r"\1[REDACTED]", s, flags=re.IGNORECASE)
    return " ".join(s.split()).strip()


def classify_provider_error(exc: Exception) -> Tuple[str, Optional[int], bool, bool]:
    """Classify an LLM provider exception into error characteristics.

    Returns:
        (category, status_code, is_retryable_on_same_provider, can_failover)
    """
    if isinstance(exc, ProviderAuthError):
        status_code = getattr(exc, "status_code", 401) or 401
        return ("auth_error", status_code, False, True)

    if isinstance(exc, RateLimitError):
        return ("rate_limit", 429, True, True)

    if isinstance(exc, ProviderTimeoutError) or isinstance(exc, httpx.TimeoutException):
        status_code = getattr(exc, "status_code", 408) or 408
        return ("timeout", status_code, False, True)

    if isinstance(exc, httpx.RequestError) or isinstance(exc, ConnectionError):
        return ("network_error", None, True, True)

    status_code: Optional[int] = getattr(exc, "status_code", None)
    if status_code is None:
        msg = str(exc)
        match = re.search(r"\bHTTP\s*(\d{3})\b", msg, re.IGNORECASE)
        if match:
            status_code = int(match.group(1))
        elif "503" in msg or "overloaded" in msg.lower() or "service unavailable" in msg.lower():
            status_code = 503
        elif "502" in msg or "bad gateway" in msg.lower():
            status_code = 502
        elif "504" in msg or "gateway timeout" in msg.lower():
            status_code = 504
        elif "429" in msg or "rate limit" in msg.lower() or "quota" in msg.lower():
            status_code = 429
        elif "401" in msg or "403" in msg or "unauthorized" in msg.lower() or "forbidden" in msg.lower():
            status_code = 401
        elif "400" in msg or "bad request" in msg.lower():
            status_code = 400

    if status_code in AUTH_STATUS_CODES:
        return ("auth_error", status_code, False, True)
    elif status_code == 429:
        return ("rate_limit", 429, True, True)
    elif status_code in (500, 502, 503, 504):
        return ("server_error", status_code, True, True)
    elif status_code in NON_RETRYABLE_CLIENT_CODES:
        return ("bad_request", status_code, False, False)
    elif status_code == 404:
        return ("model_not_found", 404, False, True)
    elif isinstance(exc, ProviderError):
        return ("provider_error", status_code, False, True)
    else:
        return ("unexpected_error", status_code, False, True)


def is_provider_configured(provider_type: str) -> bool:
    """Check if a provider has required environment credentials configured."""
    name = (provider_type or "").split(":")[0].lower().strip()
    if name == "mock":
        return True
    elif name == "gemini":
        return bool(os.environ.get("GEMINI_API_KEY", "").strip())
    elif name == "openai":
        return bool(os.environ.get("OPENAI_API_KEY", "").strip())
    elif name == "groq":
        return bool(os.environ.get("GROQ_API_KEY", "").strip())
    elif name == "ollama":
        return bool(os.environ.get("OLLAMA_BASE_URL", "").strip() or os.environ.get("OLLAMA_MODEL", "").strip())
    return False


def get_default_fallback_providers(primary_name: str) -> List[str]:
    """Return ordered list of configured fallback providers, excluding primary."""
    candidates = ["groq", "openai", "gemini", "ollama"]
    configured_fallbacks = []
    for cand in candidates:
        if cand == primary_name.lower():
            continue
        if is_provider_configured(cand):
            configured_fallbacks.append(cand)
    return configured_fallbacks


def resolve_fallback_chain(
    primary_provider: BaseLLMProvider,
    fallback_configs: Optional[List[Union[BaseLLMProvider, str]]] = None,
) -> List[BaseLLMProvider]:
    """Resolve and instantiate the ordered fallback provider chain.

    Resolution:
    1. If fallback_configs is explicitly provided (list of providers or strings), use it.
    2. Else if LLM_FALLBACK_PROVIDERS env var is set, parse comma-separated string.
    3. Else discover any configured providers among supported candidates (groq, openai, ollama).

    Skips providers with missing credentials safely.
    """
    chain: List[BaseLLMProvider] = []

    # 1. Determine candidate fallback specifications
    candidates: List[Union[BaseLLMProvider, str]] = []
    if fallback_configs is not None:
        candidates = list(fallback_configs)
    else:
        env_fallbacks = os.environ.get("LLM_FALLBACK_PROVIDERS")
        if env_fallbacks:
            candidates = [p.strip() for p in env_fallbacks.split(",") if p.strip()]
        else:
            candidates = []

    # 2. Instantiate and validate each candidate
    for item in candidates:
        if isinstance(item, BaseLLMProvider):
            if item is not primary_provider and item not in chain:
                chain.append(item)
            continue

        if not isinstance(item, str):
            continue

        raw = item.strip()
        if not raw:
            continue

        ptype, _, mname = raw.partition(":")
        ptype = ptype.strip().lower()
        mname = mname.strip() if mname else None

        # Avoid adding identical provider + model as primary
        if ptype == primary_provider.name.lower() and (
            mname is None or mname == getattr(primary_provider, "default_model", None)
        ):
            continue

        if not is_provider_configured(ptype):
            logger.info(f"Skipping unconfigured fallback provider '{ptype}'")
            continue

        try:
            prov = get_llm_provider(provider_type=ptype, model_name=mname)
            chain.append(prov)
        except Exception as e:
            logger.warning(f"Could not initialize fallback provider '{ptype}': {e}")

    return chain
