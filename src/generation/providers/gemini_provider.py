"""Google Gemini LLM provider using direct REST API via httpx."""

import os
import time
from typing import List, Dict, Any, Optional
import httpx

from src.generation.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    ProviderError,
    ProviderAuthError,
    ProviderTimeoutError,
    RateLimitError,
)


class GeminiProvider(BaseLLMProvider):
    """Provider connecting to Google Gemini REST API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gemini-flash-lite-latest",
        default_timeout: float = 30.0,
    ) -> None:
        """Initialize Gemini provider.

        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.
            default_model: Default model identifier (defaults to 'gemini-flash-lite-latest' or GEMINI_MODEL env var).
            default_timeout: Default request timeout in seconds.
        """
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._default_model = os.environ.get("GEMINI_MODEL") or default_model
        self._default_timeout = default_timeout
        self._base_url = "https://generativelanguage.googleapis.com/v1beta"

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def default_model(self) -> str:
        return self._default_model

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: Optional[float] = None,
        response_format: Optional[str] = None,
    ) -> LLMResponse:
        """Execute generateContent call against Google Gemini API."""
        if not self._api_key:
            raise ProviderAuthError(
                "Gemini API key is missing. Set the GEMINI_API_KEY environment variable "
                "or pass api_key explicitly."
            )

        target_model = model or self._default_model
        req_timeout = timeout or self._default_timeout

        # Separate system instruction from conversation contents
        system_instruction_parts = []
        gemini_contents = []

        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            if role == "system":
                system_instruction_parts.append({"text": text})
            elif role == "user":
                gemini_contents.append({"role": "user", "parts": [{"text": text}]})
            elif role == "assistant":
                gemini_contents.append({"role": "model", "parts": [{"text": text}]})

        payload: Dict[str, Any] = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        if system_instruction_parts:
            payload["systemInstruction"] = {"parts": system_instruction_parts}

        if response_format == "json_object":
            payload["generationConfig"]["responseMimeType"] = "application/json"

        url = f"{self._base_url}/models/{target_model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._api_key,
        }
        start_time = time.perf_counter()

        try:
            with httpx.Client(timeout=req_timeout) as client:
                response = client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(f"Gemini API request timed out after {req_timeout}s: {e}") from e
        except httpx.RequestError as e:
            raise ProviderError(f"Gemini connection error: {e}") from e

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if response.status_code in (401, 403):
            raise ProviderAuthError(
                f"Gemini authentication failed: {response.text}",
                status_code=response.status_code,
            )
        elif response.status_code == 429:
            response_headers = dict(getattr(response, "headers", {}))
            retry_after = response_headers.get("retry-after")
            try:
                retry_after_sec = float(retry_after) if retry_after else None
            except ValueError:
                retry_after_sec = None
            raise RateLimitError(
                f"Gemini rate limit exceeded: {response.text}",
                status_code=429,
                retry_after_seconds=retry_after_sec,
                headers=response_headers,
            )
        elif response.status_code >= 400:
            raise ProviderError(
                f"Gemini API returned error HTTP {response.status_code}: {response.text}",
                status_code=response.status_code,
            )

        try:
            res_json = response.json()
            candidates = res_json.get("candidates", [])
            if not candidates:
                # Handle cases where model blocked prompt (e.g., promptFeedback.blockReason)
                block_reason = res_json.get("promptFeedback", {}).get("blockReason", "unknown")
                raise ProviderError(f"Gemini returned no candidates (blocked: {block_reason})")
            candidate = candidates[0]
            content_obj = candidate.get("content") or {}
            parts = content_obj.get("parts", [])
            content = "".join(p.get("text", "") for p in parts)
            usage = res_json.get("usageMetadata", {})
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"Failed to parse Gemini response: {e}") from e

        return LLMResponse(
            content=content,
            model=target_model,
            prompt_tokens=usage.get("promptTokenCount"),
            completion_tokens=usage.get("candidatesTokenCount"),
            total_tokens=usage.get("totalTokenCount"),
            latency_ms=round(elapsed_ms, 2),
            raw_response=res_json,
        )

