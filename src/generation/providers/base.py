"""Base interfaces and exception classes for LLM provider abstraction."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class GenerationError(Exception):
    """Base exception for all generation and provider errors."""
    pass


class ProviderError(GenerationError):
    """Raised when an external LLM API reports an error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
        response_body: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers or {}
        self.response_body = response_body


class ProviderAuthError(ProviderError):
    """Raised when API credentials are missing, rejected, or expired."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
        response_body: Optional[str] = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code or 401,
            headers=headers,
            response_body=response_body,
        )


class ProviderTimeoutError(ProviderError):
    """Raised when an LLM provider request exceeds the timeout limit."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
        response_body: Optional[str] = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code or 408,
            headers=headers,
            response_body=response_body,
        )


class RateLimitError(ProviderError):
    """Raised when the provider rate limit or quota is exceeded."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = 429,
        retry_after_seconds: Optional[float] = None,
        reset_tokens_seconds: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
        limit_kind: Optional[str] = None,
    ) -> None:
        super().__init__(message, status_code=status_code or 429, headers=headers)
        self.retry_after_seconds = retry_after_seconds
        self.reset_tokens_seconds = reset_tokens_seconds
        self.limit_kind = limit_kind

    @property
    def retry_delay_seconds(self) -> Optional[float]:
        """Return the strongest provider-provided delay signal, if available."""
        values = [value for value in (self.retry_after_seconds, self.reset_tokens_seconds) if value is not None]
        return max(values) if values else None


class MalformedResponseError(GenerationError):
    """Raised when the provider returns a response that cannot be parsed."""
    pass


class LLMResponse(BaseModel):
    """Standardized response container across all LLM providers."""
    model_config = ConfigDict(extra="ignore")

    content: str = Field(description="Raw text or JSON content returned by the LLM")
    model: str = Field(description="Model identifier that produced the completion")
    prompt_tokens: Optional[int] = Field(default=None, description="Prompt token count")
    completion_tokens: Optional[int] = Field(default=None, description="Completion token count")
    total_tokens: Optional[int] = Field(default=None, description="Total tokens used")
    latency_ms: float = Field(default=0.0, description="Roundtrip API latency in milliseconds")
    raw_response: Optional[Dict[str, Any]] = Field(default=None, description="Raw provider response payload")


class BaseLLMProvider(ABC):
    """Abstract interface defining required methods for switchable LLM backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable identifier of the provider (e.g. 'openai', 'gemini', 'mock')."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Return the default model name for this provider."""
        pass

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: Optional[float] = None,
        response_format: Optional[str] = None,
    ) -> LLMResponse:
        """Execute a completion call against the provider.

        Args:
            messages: List of chat messages with 'role' ('system'/'user'/'assistant') and 'content'.
            model: Optional model override.
            temperature: Sampling temperature.
            max_tokens: Token generation limit.
            timeout: Request timeout override in seconds.
            response_format: Optional format constraint (e.g. 'json_object').

        Returns:
            LLMResponse containing generated content and metadata.

        Raises:
            ProviderAuthError: If authentication fails.
            ProviderTimeoutError: If the request times out.
            RateLimitError: If rate limit is hit.
            ProviderError: If the API returns any other error.
        """
        pass

