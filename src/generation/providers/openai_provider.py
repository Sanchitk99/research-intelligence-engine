"""OpenAI-compatible LLM provider using direct HTTP requests via httpx."""

import os
import json
import re
import time
from typing import List, Dict, Any, Optional, Mapping, Tuple, Union
import httpx

from src.generation.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    ProviderError,
    ProviderAuthError,
    ProviderTimeoutError,
    RateLimitError,
)


class OpenAIProvider(BaseLLMProvider):
    """Provider connecting to OpenAI or any OpenAI-compatible chat completions API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "gpt-4o-mini",
        default_timeout: float = 30.0,
        provider_name: str = "openai",
    ) -> None:
        """Initialize OpenAI-compatible provider.

        Args:
            api_key: Provider API key. If None, reads from OPENAI_API_KEY environment variable.
            base_url: Base API URL (defaults to https://api.openai.com/v1 or OPENAI_BASE_URL env var).
            default_model: Default model identifier.
            default_timeout: Default request timeout in seconds.
            provider_name: Name identifier for provider (e.g. 'openai', 'ollama').
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self._default_model = default_model
        self._default_timeout = default_timeout
        self._provider_name = provider_name

    @property
    def name(self) -> str:
        return self._provider_name

    @property
    def default_model(self) -> str:
        return self._default_model

    @staticmethod
    def _duration_seconds(value: Optional[str]) -> Optional[float]:
        """Parse Groq reset values such as ``2s``, ``250ms``, or ``1m``."""
        if value is None:
            return None
        text = str(value).strip().lower()
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(ms|s|m|h)?", text)
        if not match:
            return None
        number = float(match.group(1))
        multiplier = {None: 1.0, "ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[match.group(2)]
        return number * multiplier

    @classmethod
    def _rate_limit_metadata(cls, headers: Mapping[str, str], body: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        normalized = {str(key).lower(): str(value) for key, value in headers.items()}
        retry_after = cls._duration_seconds(normalized.get("retry-after"))
        reset_tokens = cls._duration_seconds(normalized.get("x-ratelimit-reset-tokens"))
        limit_kind = None
        try:
            payload = json.loads(body)
            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            message = str(error.get("message", ""))
            error_type = str(error.get("type", ""))
            if "token" in f"{message} {error_type}".lower() or reset_tokens is not None:
                limit_kind = "tokens"
            elif error_type or message:
                limit_kind = "requests"
        except (TypeError, ValueError):
            pass
        if limit_kind is None and reset_tokens is not None:
            limit_kind = "tokens"
        return retry_after, reset_tokens, limit_kind

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_completion_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        response_format: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Execute chat completion request against OpenAI API."""
        display_name = {"ollama": "Ollama", "groq": "Groq"}.get(self.name, "OpenAI")
        if not self._api_key:
            raise ProviderAuthError(
                f"{display_name} API key is missing. Set the {display_name.upper()}_API_KEY environment variable "
                "or pass api_key explicitly."
            )

        target_model = model or self._default_model
        req_timeout = timeout or self._default_timeout

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_completion_tokens is not None:
            payload["max_completion_tokens"] = max_completion_tokens
        else:
            payload["max_tokens"] = max_tokens

        if isinstance(response_format, dict):
            payload["response_format"] = response_format
        elif response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}

        url = f"{self._base_url}/chat/completions"
        start_time = time.perf_counter()

        try:
            with httpx.Client(timeout=req_timeout) as client:
                response = client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            raise ProviderTimeoutError(f"{display_name} API request timed out after {req_timeout}s: {e}") from e
        except httpx.RequestError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            raise ProviderError(f"{display_name} connection error: {e}") from e

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if response.status_code in (401, 403):
            raise ProviderAuthError(
                f"{display_name} authentication failed: {response.text}",
                status_code=response.status_code,
            )
        elif response.status_code == 429:
            response_headers = getattr(response, "headers", {})
            retry_after, reset_tokens, limit_kind = self._rate_limit_metadata(response_headers, response.text)
            metadata = {
                "retry-after": response_headers.get("retry-after"),
                "x-ratelimit-remaining-tokens": response_headers.get("x-ratelimit-remaining-tokens"),
                "x-ratelimit-reset-tokens": response_headers.get("x-ratelimit-reset-tokens"),
            }
            safe_headers = {key: value for key, value in metadata.items() if value is not None}
            raise RateLimitError(
                f"{display_name} rate limit exceeded: {response.text}; headers={safe_headers}",
                status_code=429,
                retry_after_seconds=retry_after,
                reset_tokens_seconds=reset_tokens,
                headers=safe_headers,
                limit_kind=limit_kind,
            )
        elif response.status_code >= 400:
            raise ProviderError(
                f"{display_name} API returned error HTTP {response.status_code}: {response.text}",
                status_code=response.status_code,
            )

        try:
            res_json = response.json()
            choices = res_json.get("choices", [])
            if not choices:
                raise ProviderError(f"{display_name} returned no choices: {res_json}")
            choice = choices[0]
            message_obj = choice.get("message") or {}
            content = message_obj.get("content") or message_obj.get("refusal") or ""
            usage = res_json.get("usage", {})
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"Failed to parse {display_name} response: {e}") from e

        return LLMResponse(
            content=content,
            model=target_model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            latency_ms=round(elapsed_ms, 2),
            raw_response=res_json,
        )

