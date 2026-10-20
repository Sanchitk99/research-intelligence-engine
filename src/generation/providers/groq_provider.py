"""Groq provider using Groq's OpenAI-compatible chat-completions API."""

import os
import threading
import time
from typing import Optional

from src.generation.providers.base import RateLimitError
from src.generation.providers.openai_provider import OpenAIProvider


GROQ_SCHEMA_VERSION = "research-intelligence-answer-v1"
GROQ_DEFAULT_MAX_COMPLETION_TOKENS = 2048
GROQ_REASONING_EFFORT = "medium"  # GPT-OSS provider default; not sent explicitly.
GROQ_REASONING_FORMAT = None  # Groq documents reasoning_format as unsupported for GPT-OSS.


GROQ_GENERATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": [
                "answered",
                "partially_answered",
                "abstained_out_of_scope",
                "abstained_insufficient_evidence",
                "conflicting_evidence",
            ],
        },
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "excerpt": {"type": "string"},
                },
                "required": ["chunk_id", "excerpt"],
                "additionalProperties": False,
            },
        },
        "confidence_notes": {"type": ["string", "null"]},
    },
    "required": ["status", "answer", "citations", "confidence_notes"],
    "additionalProperties": False,
}


def groq_generation_response_format() -> dict:
    """Return the strict Groq schema used by AnswerGenerator JSON responses."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "research_intelligence_answer",
            "strict": True,
            "schema": GROQ_GENERATION_JSON_SCHEMA,
        },
    }


class GroqProvider(OpenAIProvider):
    """Provider connecting to Groq through its official OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        default_timeout: float = 30.0,
    ) -> None:
        super().__init__(
            api_key=api_key or os.environ.get("GROQ_API_KEY"),
            base_url=base_url or os.environ.get("GROQ_BASE_URL") or "https://api.groq.com/openai/v1",
            default_model=default_model or os.environ.get("GROQ_MODEL") or "openai/gpt-oss-20b",
            default_timeout=default_timeout,
            provider_name="groq",
        )

        self._max_rate_limit_retries = max(0, int(os.environ.get("GROQ_MAX_RATE_LIMIT_RETRIES", "2")))
        self._rate_limit_buffer_seconds = max(0.0, float(os.environ.get("GROQ_RATE_LIMIT_BUFFER_SECONDS", "0.5")))
        self._max_completion_tokens = int(
            os.environ.get("GROQ_MAX_COMPLETION_TOKENS", str(GROQ_DEFAULT_MAX_COMPLETION_TOKENS))
        )
        if self._max_completion_tokens < 64 or self._max_completion_tokens > 65536:
            raise ValueError("GROQ_MAX_COMPLETION_TOKENS must be between 64 and 65536")
        self._generation_lock = threading.Lock()

    @property
    def max_completion_tokens(self) -> int:
        return self._max_completion_tokens

    def generate(self, *args, **kwargs):
        """Generate with bounded, header-directed retries for Groq 429 responses."""
        if kwargs.get("response_format") == "json_object":
            kwargs["response_format"] = groq_generation_response_format()
        if "max_completion_tokens" not in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens", self._max_completion_tokens)
        with self._generation_lock:
            retries = 0
            while True:
                try:
                    return super().generate(*args, **kwargs)
                except RateLimitError as exc:
                    if retries >= self._max_rate_limit_retries:
                        raise RateLimitError(
                            f"Groq rate limit retry budget exhausted after {retries} retries: {exc}",
                            retry_after_seconds=exc.retry_after_seconds,
                            reset_tokens_seconds=exc.reset_tokens_seconds,
                            headers=exc.headers,
                            limit_kind=exc.limit_kind,
                        ) from exc
                    delay = exc.retry_delay_seconds
                    if delay is None:
                        raise RateLimitError(
                            f"Groq rate limit response did not include retry/reset timing; refusing to retry: {exc}",
                            headers=exc.headers,
                            limit_kind=exc.limit_kind,
                        ) from exc
                    retries += 1
                    time.sleep(delay + self._rate_limit_buffer_seconds)

